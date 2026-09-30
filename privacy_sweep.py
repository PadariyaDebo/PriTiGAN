"""
    python privacy_sweep.py --dataset stock \
        --noise_mults 0.5 0.8 1.0 1.5 3.0 5.0 10.0 20.0 50.0 90.0
"""

import argparse
import csv
from pathlib import Path

from PriTi.data  import load_dataset, preprocess, DATASET_CONFIG
from PriTi.model import estimate_epsilon


def sweep(dataset_name: str,
         noise_mults:   list,
         iterations:    int = 10_000,
         delta:         float = 1e-5,
         td_fraction:   float = 0.5,
         data_path:     str = None,
         output_dir:    str = "outputs"):
    cfg = DATASET_CONFIG[dataset_name]
    seq_len    = cfg["seq_len"]
    batch_size = cfg["batch_size"]

    print(f"Loading '{dataset_name}' to determine real n_train ...")
    df = load_dataset(dataset_name, data_path)
    train_seq, _test_seq, _scaler, _train_scaled, _test_scaled = preprocess(
        df, seq_len, train_ratio=0.70, seed=42)
    n_train = len(train_seq)
    print(f"  n_train = {n_train}, batch_size = {batch_size}, "
          f"q = {batch_size / n_train:.5f}")

    # embedding updates twice per outer step (see train.py)
    t_embedding = 2 * iterations
    t_disc_upper = iterations
    t_disc_frac  = int(td_fraction * iterations)

    # num_microbatches = 1 in train.py: the batch gradient is clipped as a
    # whole, so the sensitivity is 2C and the effective noise multiplier for
    # accounting is noise_multiplier / 2 (same as train.py).
    rows = []
    for nm in noise_mults:
        eps_upper = estimate_epsilon(
            n_train=n_train, batch_size=batch_size, noise_multiplier=nm / 2.0,  # one microbatch: sensitivity 2C
            t_embedding=t_embedding, t_discriminator=t_disc_upper, delta=delta)
        eps_frac = estimate_epsilon(
            n_train=n_train, batch_size=batch_size, noise_multiplier=nm / 2.0,  # one microbatch: sensitivity 2C
            t_embedding=t_embedding, t_discriminator=t_disc_frac, delta=delta)
        rows.append((nm, eps_upper, eps_frac))
        print(f"  noise_multiplier={nm:>7.3f}  |  "
              f"eps (T_d=100% of iters) = {eps_upper:9.4f}  |  "
              f"eps (T_d={int(td_fraction*100)}% of iters) = {eps_frac:9.4f}")

    out = Path(output_dir) / dataset_name
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "privacy_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["noise_multiplier",
                    f"epsilon_Td={t_disc_upper}(100pct)",
                    f"epsilon_Td={t_disc_frac}({int(td_fraction*100)}pct)"])
        w.writerows(rows)
    print(f"\nSaved sweep table -> {csv_path}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep noise_multiplier -> epsilon for a PriTiGAN dataset")
    parser.add_argument("--dataset",     type=str,   default="stock",
                        choices=["stock", "energy", "mba"])
    parser.add_argument("--data_path",   type=str,   default=None)
    parser.add_argument("--iterations",  type=int,   default=10_000)
    parser.add_argument("--delta",       type=float, default=1e-5)
    parser.add_argument("--td_fraction", type=float, default=0.5,
                        help="Fraction of iterations to use as a second "
                             "T_d reference point (default: 0.5)")
    parser.add_argument("--output_dir",  type=str,   default="outputs")
    parser.add_argument("--noise_mults", type=float, nargs="+",
                        default=[0.5, 0.8, 1.0, 1.5, 3.0, 5.0,
                                 10.0, 20.0, 50.0, 90.0])
    args = parser.parse_args()

    sweep(
        dataset_name=args.dataset,
        noise_mults=args.noise_mults,
        iterations=args.iterations,
        delta=args.delta,
        td_fraction=args.td_fraction,
        data_path=args.data_path,
        output_dir=args.output_dir,
    )
    
