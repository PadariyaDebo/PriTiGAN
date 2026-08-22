"""
Trains a DoppelGANger model (Lin et al., 2020) on the same preprocessed
data PriTiGAN uses, so results are directly comparable.

*** RUN THIS IN ITS OWN VENV, SEPARATE FROM THE MAIN REPO ***

ydata-synthetic's DoppelGANger calls tf.compat.v1.disable_eager_execution()
internally, which switches the whole TF runtime to graph mode for the rest
of the process -- incompatible with PriTiGAN's eager-mode @tf.function
training code. Don't import this from the same process as pritigan.model
or train.py.

    python3.10 -m venv venv_doppelganger
    source venv_doppelganger/bin/activate
    pip install -r baselines/requirements-doppelganger.txt
    python baselines/doppelganger_baseline.py --dataset stock

Produces outputs/<dataset>/doppelganger/synthetic_sequences.npy in the same
(N, seq_len, n_features) format train.py produces, so
baselines/evaluate_baseline.py (numpy only, no ydata-synthetic needed) can
compare them with the same pipeline.

Data format note: PriTiGAN trains on overlapping sliding windows
(stride=1), but DoppelGANger's .fit() expects one continuous series that it
chunks itself into non-overlapping sequence_length blocks. Feeding it
PriTiGAN's windowed data (flattened back out) would just hand it a heavily
duplicated signal, so this script uses preprocess()'s train_scaled /
test_scaled (continuous, normalised, not yet windowed) instead.

Only black-box MIA applies to this baseline -- see evaluate_baseline.py.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pritigan.data import load_dataset, preprocess, DATASET_CONFIG


def train_doppelganger(dataset_name: str,
                       data_path:    str = None,
                       epochs:       int = 400,
                       sample_length: int = 1,
                       output_dir:   str = "outputs",
                       seed:         int = 42):
    try:
        from ydata_synthetic.synthesizers.timeseries import TimeSeriesSynthesizer
        from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
    except ImportError as e:
        raise ImportError(
            "ydata-synthetic isn't installed, or is installed in the wrong "
            "environment -- this script needs its own venv, see the module "
            "docstring. pip install -r baselines/requirements-doppelganger.txt"
        ) from e

    np.random.seed(seed)

    cfg = DATASET_CONFIG[dataset_name]
    seq_len = cfg["seq_len"]

    print(f"[1/4] Loading '{dataset_name}' with the same preprocessing "
          f"PriTiGAN uses...")
    df = load_dataset(dataset_name, data_path)
    col_names = list(df.columns)
    train_seq, test_seq, scaler, train_scaled, test_scaled = preprocess(
        df, seq_len, train_ratio=0.70, seed=seed)

    if seq_len % sample_length != 0:
        raise ValueError(f"--sample_length={sample_length} must divide "
                         f"seq_len={seq_len}. Choose a divisor of {seq_len}.")

    print(f"[2/4] Building continuous training DataFrame "
          f"({train_scaled.shape[0]} rows, {train_scaled.shape[1]} columns)...")
    # DGAN chunks this into non-overlapping seq_len blocks itself -- don't
    # pass PriTiGAN's overlapping windowed train_seq here.
    n_usable_rows = (len(train_scaled) // seq_len) * seq_len
    train_df = pd.DataFrame(train_scaled[:n_usable_rows], columns=col_names)

    print(f"[3/4] Training DoppelGANger "
          f"(epochs={epochs}, sequence_length={seq_len}, "
          f"sample_length={sample_length})...")
    model_args = ModelParameters(batch_size=min(cfg["batch_size"], len(train_df) // seq_len))
    train_args = TrainParameters(
        epochs=epochs,
        sequence_length=seq_len,
        sample_length=sample_length,
        rounds=1,
        measurement_cols=col_names,
    )
    model = TimeSeriesSynthesizer(modelname="doppelganger",
                                  model_parameters=model_args)
    # data.py already dropped non-numeric columns, so there are no
    # categorical/static attribute columns here -- PriTiGAN and
    # DoppelGANger see the same input information. (DoppelGANger can
    # natively condition on categorical attributes, e.g. MBA's isp/
    # technology/state columns; we skip that here for a fair comparison
    # against PriTiGAN, which can't use them.)
    model.fit(train_df, train_args, num_cols=col_names, cat_cols=[])

    n_synth_sequences = len(train_seq)
    print(f"[4/4] Sampling {n_synth_sequences} synthetic sequences...")
    synth_list = model.sample(n_samples=n_synth_sequences)
    # .sample() returns a list of per-sequence DataFrames, not a flat array
    # (matches the documented usage pattern of pd.concat(synth_data, axis=0))
    synth_seq = np.stack(
        [d[col_names].values.astype(np.float32) for d in synth_list], axis=0)
    # fail loudly if DGAN's internal chunking returned a different shape
    # than requested, rather than silently mismatching downstream
    if synth_seq.shape[1:] != (seq_len, len(col_names)):
        raise ValueError(
            f"DoppelGANger returned sequences of shape {synth_seq.shape[1:]}, "
            f"expected ({seq_len}, {len(col_names)}). Check sample_length / "
            f"sequence_length configuration.")

    out = Path(output_dir) / dataset_name / "doppelganger"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "train_sequences.npy", train_seq)
    np.save(out / "test_sequences.npy",  test_seq)
    np.save(out / "synthetic_sequences.npy", synth_seq)
    model.save(str(out / "model"))
    print(f"  Saved synthetic data + model -> {out}/")
    print(f"\n  Next: run baselines/evaluate_baseline.py --dataset {dataset_name} "
          f"--baseline doppelganger (in the MAIN repo's environment, not "
          f"this one -- that script only needs numpy, not ydata-synthetic).")

    return synth_seq


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a DoppelGANger baseline (separate env required, see module docstring)")
    parser.add_argument("--dataset",       type=str, default="stock",
                        choices=["stock", "energy", "mba"])
    parser.add_argument("--data_path",     type=str, default=None)
    parser.add_argument("--epochs",        type=int, default=400,
                        help="DoppelGANger epochs (not directly comparable to "
                             "PriTiGAN's --iterations; tune for convergence)")
    parser.add_argument("--sample_length", type=int, default=1,
                        help="Must divide the dataset's seq_len "
                             "(24 for stock/energy, 56 for mba)")
    parser.add_argument("--output_dir",    type=str, default="outputs")
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    train_doppelganger(
        dataset_name=args.dataset,
        data_path=args.data_path,
        epochs=args.epochs,
        sample_length=args.sample_length,
        output_dir=args.output_dir,
        seed=args.seed,
    )
