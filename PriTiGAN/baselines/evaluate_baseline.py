"""
Evaluates a saved synthetic_sequences.npy (from any baseline) against real
data with the same metrics evaluate.py uses. Runs in the main repo's
environment -- doesn't need ydata-synthetic or anything TF1.

Usage (after doppelganger_baseline.py has produced
outputs/<dataset>/doppelganger/*.npy in its own env):

    python baselines/evaluate_baseline.py --dataset stock --baseline doppelganger

Only black-box (Monte Carlo) MIA runs here. White-box discriminator MIA
needs a live TF2-eager discriminator, which external baselines like
DoppelGANger (TF1 graph mode) don't expose. Since we only have a fixed
array of saved synthetic samples rather than a callable generator, the
generator_fn for Monte Carlo MIA resamples with replacement from that
array -- an approximation of sampling from a live generator.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.metrics import evaluate_all
from mia.attacks import monte_carlo_mia


def evaluate_baseline(dataset_name: str,
                      baseline:     str,
                      output_dir:   str = "outputs",
                      n_mia_runs:   int = 5,
                      member_size:  int = 500,
                      seed:         int = 42):
    out = Path(output_dir) / dataset_name / baseline

    print(f"Loading data from {out}/ ...")
    train_seq = np.load(out / "train_sequences.npy")
    test_seq  = np.load(out / "test_sequences.npy")
    synth     = np.load(out / "synthetic_sequences.npy")
    print(f"  Train:     {train_seq.shape}")
    print(f"  Test:      {test_seq.shape}")
    print(f"  Synthetic: {synth.shape}")

    n_eval = min(len(test_seq), len(synth))
    results = evaluate_all(
        real=test_seq[:n_eval],
        synth=synth[:n_eval],
        dataset_name=f"{dataset_name} [{baseline}]",
        run_downstream=True,
        seed=seed,
    )

    # black-box MIA only, see module docstring
    print(f"\nBlack-box MIA ({n_mia_runs} runs, resampling from saved "
          f"synthetic data as an approximate generator_fn)...")

    def resample_generator_fn(n_samples):
        idx = np.random.choice(len(synth), size=n_samples, replace=True)
        return synth[idx]

    bb_aucs = []
    for run in range(n_mia_runs):
        rng = np.random.default_rng(seed + run)
        n_m  = min(member_size, len(train_seq))
        n_nm = min(member_size, len(test_seq))
        members     = train_seq[rng.permutation(len(train_seq))[:n_m]].astype(np.float32)
        non_members = test_seq[rng.permutation(len(test_seq))[:n_nm]].astype(np.float32)

        auc = monte_carlo_mia(
            generator_fn=resample_generator_fn,
            member_seqs=members,
            non_member_seqs=non_members,
            n_mc_samples=1000,
            n_pca_components=5,
            seed=seed + run,
        )
        bb_aucs.append(auc)
        print(f"  Run {run + 1}/{n_mia_runs} | BB-AUC: {auc:.3f}")

    results["bb_auc_mean"] = float(np.mean(bb_aucs))
    results["bb_auc_std"]  = float(np.std(bb_aucs))
    print(f"\n  BB AUC: {results['bb_auc_mean']:.3f} ± {results['bb_auc_std']:.3f}")
    print(f"  White-box MIA: NOT RUN (see module docstring -- not applicable "
          f"to external baselines without a TF2-eager discriminator)")

    print("\n" + "="*50)
    print(f" SUMMARY — {dataset_name} [{baseline}]")
    print("="*50)
    print(f"  JSD mean          : {results.get('jsd_mean', float('nan')):.4f}")
    print(f"  WD  mean          : {results.get('wd_mean',  float('nan')):.4f}")
    print(f"  DTW mean          : {results.get('dtw',      float('nan')):.4f}")
    print(f"  Classification Acc: {results.get('classification_accuracy', 0)*100:.2f}%")
    print(f"  Regression R²     : {results.get('r2',  float('nan')):.4f}")
    print(f"  Regression MAE    : {results.get('mae', float('nan')):.4f}")
    print(f"  BB-AUC (MIA)      : {results['bb_auc_mean']:.3f} ± {results['bb_auc_std']:.3f}")
    print("="*50)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a saved baseline's synthetic data (main repo env)")
    parser.add_argument("--dataset",     type=str, default="stock",
                        choices=["stock", "energy", "mba"])
    parser.add_argument("--baseline",    type=str, default="doppelganger")
    parser.add_argument("--output_dir",  type=str, default="outputs")
    parser.add_argument("--n_mia_runs",  type=int, default=5)
    parser.add_argument("--member_size", type=int, default=500)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    evaluate_baseline(
        dataset_name=args.dataset,
        baseline=args.baseline,
        output_dir=args.output_dir,
        n_mia_runs=args.n_mia_runs,
        member_size=args.member_size,
        seed=args.seed,
    )
