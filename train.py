"""
--baseline flag selects which networks get DP noise:
  pritigan       (default) -- both embedding and discriminator
  dptimegan      -- discriminator only
  embedding_only -- embedding only
  nondp          -- neither (TimeGAN)
"""

import argparse
import csv
import json
import time
import numpy as np
import tensorflow as tf
from pathlib import Path

from PriTi.model import PriTiGAN, estimate_epsilon
from PriTi.data  import load_dataset, preprocess, DATASET_CONFIG

# Default noise_multiplier / l2_norm_clip per dataset
DEFAULT_DP = {
    "stock":  {"noise_multiplier": 90.0, "l2_norm_clip": 1.5},
    "energy": {"noise_multiplier": 1.0,  "l2_norm_clip": 1.0},
    "mba":    {"noise_multiplier": 0.8,  "l2_norm_clip": 1.0},
}

# --baseline -> (dp_embedding, dp_discriminator)
BASELINE_DP_FLAGS = {
    "pritigan":       (True,  True),
    "dptimegan":      (False, True),
    "embedding_only": (True,  False),
    "nondp":          (False, False),
}


def make_tf_dataset(sequences: np.ndarray,
                    batch_size: int,
                    seed: int = 42):
    """Create a repeating, shuffled tf.data pipeline."""
    ds = (tf.data.Dataset
          .from_tensor_slices(sequences)
          .shuffle(buffer_size=len(sequences), seed=seed)
          .batch(batch_size, drop_remainder=True)
          .repeat())
    return iter(ds)


def make_noise_iter(batch_size: int, seq_len: int, n_features: int):
    """Infinite generator of uniform noise batches (Z ~ U[0,1])."""
    def _gen():
        while True:
            yield np.random.uniform(0, 1, (seq_len, n_features)).astype(np.float32)

    ds = (tf.data.Dataset
          .from_generator(_gen, output_types=tf.float32)
          .batch(batch_size, drop_remainder=True)
          .repeat())
    return iter(ds)


def train(dataset_name: str,
          data_path:    str   = None,
          iterations:   int   = 10_000,
          noise_mult:   float = None,
          l2_clip:      float = None,
          delta:        float = 1e-5,
          output_dir:   str   = "outputs",
          seed:         int   = 42,
          baseline:     str   = "pritigan"):
    """
    Three-phase training: autoencoder pretrain, supervisor pretrain, joint
    DP training, `iterations` steps each.
    """
    if baseline not in BASELINE_DP_FLAGS:
        raise ValueError(f"Unknown baseline '{baseline}'. "
                         f"Choose from {list(BASELINE_DP_FLAGS.keys())}.")
    dp_embedding, dp_discriminator = BASELINE_DP_FLAGS[baseline]

    np.random.seed(seed)
    tf.random.set_seed(seed)

    cfg = DATASET_CONFIG[dataset_name]
    seq_len    = cfg["seq_len"]
    n_features = cfg["n_features"]
    batch_size = cfg["batch_size"]

    dp_cfg = DEFAULT_DP[dataset_name]
    noise_multiplier = noise_mult if noise_mult is not None else dp_cfg["noise_multiplier"]
    l2_norm_clip     = l2_clip   if l2_clip   is not None else dp_cfg["l2_norm_clip"]

    #  Load and preprocess data 
    print(f"\n[1/4] Loading '{dataset_name}' dataset...")
    df = load_dataset(dataset_name, data_path)
    train_seq, test_seq, scaler, _train_scaled, _test_scaled = preprocess(
        df, seq_len, train_ratio=0.70, seed=seed)

    out = Path(output_dir) / dataset_name / baseline if baseline != "pritigan" \
        else Path(output_dir) / dataset_name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "test_sequences.npy",  test_seq)
    np.save(out / "train_sequences.npy", train_seq)

    #  Build model 
    print(f"\n[2/4] Building PriTiGAN (baseline={baseline}, "
          f"dp_embedding={dp_embedding}, dp_discriminator={dp_discriminator}, "
          f"noise_mult={noise_multiplier}, l2_clip={l2_norm_clip})...")
    model = PriTiGAN({
        "seq_len":          seq_len,
        "n_features":       n_features,
        "hidden_dim":       24,
        "num_layers":       3,
        "gamma":            1,
        "lambda1":          10,
        "lambda2":          0.1,
        "noise_multiplier": noise_multiplier,
        "l2_norm_clip":     l2_norm_clip,
        # one microbatch: the batch gradient is clipped as a whole (Section 5.5)
        "num_microbatches": 1,
        "learning_rate":    5e-4,
        "dp_embedding":     dp_embedding,
        "dp_discriminator": dp_discriminator,
    })

    real_iter  = make_tf_dataset(train_seq, batch_size, seed)
    noise_iter = make_noise_iter(batch_size, seq_len, n_features)

    # Phase 1: Autoencoder 
    print(f"\n[3/4] Phase 1 — Autoencoder pre-training ({iterations} steps)...")
    ae_loss_history = []
    for step in range(iterations):
        x    = next(real_iter)
        loss = model.train_autoencoder(x)
        ae_loss_history.append(float(loss))
        if step % 2000 == 0:
            print(f"  Step {step:5d} | AE loss: {float(loss):.6f}")

    #  Phase 2: Supervisor 
    print(f"\n        Phase 2 — Supervisor pre-training ({iterations} steps)...")
    sup_loss_history = []
    for step in range(iterations):
        x    = next(real_iter)
        loss = model.train_supervisor(x)
        sup_loss_history.append(float(loss))
        if step % 2000 == 0:
            print(f"  Step {step:5d} | SUP loss: {float(loss):.6f}")

    #  Phase 3: Joint training 
    print(f"\n        Phase 3 — Joint DP training ({iterations} steps)...")
    t_embedding_updates    = 0  
    t_discriminator_updates = 0  
    phase3_history = []  # step, g_u, g_s, g_v, e_loss, d_loss, d_updated
    for step in range(iterations):
        for _ in range(2):
            x = next(real_iter)
            z = next(noise_iter)
            g_u, g_s, g_v = model.train_generator(x, z)
            e_loss         = model.train_embedding_dp(x)
            t_embedding_updates += 1

        x = next(real_iter)
        z = next(noise_iter)
        d_loss = model.get_discriminator_loss(x, z)
        d_updated = False
        if d_loss > 0.15:
            d_loss = model.train_discriminator_dp(x, z)
            t_discriminator_updates += 1
            d_updated = True

        phase3_history.append((
            step, float(g_u), float(g_s), float(g_v),
            float(e_loss), float(d_loss), int(d_updated)))

        if step % 1000 == 0:
            print(f"  Step {step:5d} | "
                  f"D: {float(d_loss):.4f} | "
                  f"G_u: {float(g_u):.4f} | "
                  f"G_s: {float(g_s):.4f} | "
                  f"G_v: {float(g_v):.4f} | "
                  f"E: {float(e_loss):.4f}")

    print("\n  Saving raw training logs...")
    with open(out / "phase1_ae_loss.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "ae_loss"])
        w.writerows(enumerate(ae_loss_history))

    with open(out / "phase2_supervisor_loss.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "supervisor_loss"])
        w.writerows(enumerate(sup_loss_history))

    with open(out / "phase3_joint_loss.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "g_loss_u", "g_loss_s", "g_loss_v",
                     "embedding_loss", "discriminator_loss",
                     "discriminator_updated"])
        w.writerows(phase3_history)
    print(f"  Saved loss curves → {out}/phase{{1,2,3}}_*.csv")

    # ── Privacy budget ───────────────────────────────────────────────────────
    # only count updates from networks that were actually DP-noised in this run
    eff_t_embedding     = t_embedding_updates     if dp_embedding     else 0
    eff_t_discriminator = t_discriminator_updates if dp_discriminator else 0

    print("\n[4/4] Computing privacy budget (ε, δ)...")
    print(f"  Baseline: {baseline}  (dp_embedding={dp_embedding}, "
          f"dp_discriminator={dp_discriminator})")
    print(f"  Embedding DP updates    (T_e): {eff_t_embedding}"
          f"{'' if dp_embedding else '  [not privatized in this baseline]'}")
    print(f"  Discriminator DP updates (T_d): {eff_t_discriminator}"
          f"{'' if dp_discriminator else '  [not privatized in this baseline]'}")
    q = batch_size / len(train_seq)
    effective_noise_multiplier = noise_multiplier / 2.0
    eps = estimate_epsilon(
        n_train=len(train_seq),
        batch_size=batch_size,
        noise_multiplier=effective_noise_multiplier,
        t_embedding=eff_t_embedding,
        t_discriminator=eff_t_discriminator,
        delta=delta,
    )
    print(f"  Final ε = {eps:.4f} (δ = {delta})")

    #  Privacy accounting report
    privacy_report = {
        "dataset":                 dataset_name,
        "baseline":                baseline,
        "dp_embedding":            dp_embedding,
        "dp_discriminator":        dp_discriminator,
        "timestamp_utc":           time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_train":                 len(train_seq),
        "batch_size":              batch_size,
        "sampling_rate_q":         q,
        "noise_multiplier":        noise_multiplier,
        "num_microbatches":        1,
        "sensitivity":             "2C (batch gradient clipped as a whole)",
        "effective_noise_multiplier_for_accounting": effective_noise_multiplier,
        "l2_norm_clip":            l2_norm_clip,
        "delta":                   delta,
        "rdp_orders":              "selected internally by dp_accounting.rdp.RdpAccountant",
        "iterations":              iterations,
        "t_embedding_updates":     eff_t_embedding,
        "t_discriminator_updates": eff_t_discriminator,
        "epsilon_covers": "embedding and discriminator DP-SGD updates",
        "full_pipeline_guarantee": False,
        "not_privatised": ["autoencoder pre-training",
                           "supervisor pre-training",
                           "generator updates",
                           "discriminator update gate (d_loss > 0.15)"],
        "epsilon":                 eps,
        "seed":                    seed,
    }
    with open(out / "privacy_accounting.json", "w") as f:
        json.dump(privacy_report, f, indent=2)
    print(f"  Saved full privacy accounting report → {out / 'privacy_accounting.json'}")

    #  Generate and save synthetic data 
    print("\n  Generating synthetic sequences...")
    synth = model.generate(n_samples=len(train_seq))
    np.save(out / "synthetic_sequences.npy", synth)
    print(f"  Saved synthetic data → {out / 'synthetic_sequences.npy'}")

    # Save model weights
    model.embedder.save_weights(str(out / "embedder.weights.h5"))
    model.generator.save_weights(str(out / "generator.weights.h5"))
    model.discriminator.save_weights(str(out / "discriminator.weights.h5"))
    model.recovery.save_weights(str(out / "recovery.weights.h5"))
    model.supervisor.save_weights(str(out / "supervisor.weights.h5"))
    print(f"  Saved model weights → {out}/")

    return model, scaler, train_seq, test_seq, synth, eps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PriTiGAN (or a baseline/ablation variant) on a time-series dataset")
    parser.add_argument("--dataset",    type=str,   default="stock",
                        choices=["stock", "energy", "mba"])
    parser.add_argument("--data_path",  type=str,   default=None,
                        help="Local CSV path (optional, overrides URL)")
    parser.add_argument("--iterations", type=int,   default=10_000)
    parser.add_argument("--noise_mult", type=float, default=None,
                        help="DP noise multiplier (default: dataset-specific)")
    parser.add_argument("--l2_clip",    type=float, default=None,
                        help="DP l2 norm clip (default: dataset-specific)")
    parser.add_argument("--delta",      type=float, default=1e-5)
    parser.add_argument("--output_dir", type=str,   default="outputs")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--baseline",   type=str,   default="pritigan",
                        choices=list(BASELINE_DP_FLAGS.keys()),
                        help="pritigan (dual-noise, default) | dptimegan "
                             "(discriminator-only DP) | embedding_only "
                             "(embedding-only DP) | nondp (no DP anywhere)")
    args = parser.parse_args()

    train(
        dataset_name=args.dataset,
        data_path=args.data_path,
        iterations=args.iterations,
        noise_mult=args.noise_mult,
        l2_clip=args.l2_clip,
        delta=args.delta,
        output_dir=args.output_dir,
        seed=args.seed,
        baseline=args.baseline,
    )
