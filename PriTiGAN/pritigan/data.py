"""
Data loading + preprocessing (Section 5.2): min-max scale on train stats
only, sliding window stride=1, 70/30 split, seed=42.

Sources match the two baseline architectures this work builds on:
  - stock / energy: original TimeGAN authors' repo (jsyoon0823/TimeGAN),
    same datasets cited by the paper's [10].
  - mba: same processed CSV used by the original DoppelGANger authors'
    repo (fjxmlzn/DoppelGANger), cited by the paper's [11]. DoppelGANger's
    own repo doesn't host a directly-downloadable raw CSV (their data/
    README points to external download links instead), so this fetches the
    same file from the ydata-synthetic mirror, which redistributes it.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Optional

DATA_URLS = {
    "stock": ("https://github.com/jsyoon0823/TimeGAN/raw/refs/heads/master/"
               "data/stock_data.csv"),
    "energy": ("https://github.com/jsyoon0823/TimeGAN/raw/refs/heads/master/"
                "data/energy_data.csv"),
    "mba": ("https://raw.githubusercontent.com/ydataai/ydata-synthetic/"
             "refs/heads/dev/data/fcc_mba.csv"),
}

# Table 1 of the paper.
DATASET_CONFIG = {
    "stock": {
        "seq_len":    24,
        "n_features": 6,
        "batch_size": 128,
        # column names are matched case/whitespace-insensitively below (see
        # load_dataset), so it doesn't matter whether the source CSV uses
        # "Adj Close" or "Adj_Close" -- this is exactly the kind of naming
        # mismatch that silently dropped stock from 6 features to 5 before.
        "columns":    ["Open", "High", "Low", "Close", "Adj Close", "Volume"],
    },
    "energy": {
        "seq_len":    24,
        "n_features": 28,
        "batch_size": 128,
        "columns":    None,   # all numeric columns, date column dropped
    },
    "mba": {
        "seq_len":    56,
        # fcc_mba.csv has 5 raw columns but only 2 are numeric
        # (traffic_byte_counter, ping_loss_rate) -- isp/technology/state are
        # categorical and unused. Table 1 in the paper says 5 features, but
        # Table 5 only reports these same 2, so 2 is what's actually used.
        "n_features": 2,
        "batch_size": 100,
        "columns":    ["traffic_byte_counter", "ping_loss_rate"],
    },
}


def load_dataset(name: str, data_path: str = None) -> pd.DataFrame:
    """Load from URL, or a local path."""
    if data_path is not None:
        df = pd.read_csv(data_path)
    else:
        if name not in DATA_URLS:
            raise ValueError(f"Unknown dataset '{name}'. "
                             f"Choose from {list(DATA_URLS.keys())} "
                             f"or supply data_path.")
        df = pd.read_csv(DATA_URLS[name])

    cfg = DATASET_CONFIG[name]
    if cfg["columns"] is not None:
        # normalize before matching so "Adj Close" / "Adj_Close" / "adj
        # close" all resolve the same -- exact-match was how this silently
        # dropped a feature before when a mirror used different spacing
        def _norm(s: str) -> str:
            return s.strip().lower().replace(" ", "_")

        norm_to_actual = {_norm(c): c for c in df.columns}
        resolved, missing = [], []
        for wanted in cfg["columns"]:
            actual = norm_to_actual.get(_norm(wanted))
            if actual is None:
                missing.append(wanted)
            else:
                resolved.append(actual)

        if missing:
            raise ValueError(
                f"Dataset '{name}': requested column(s) {missing} not found "
                f"in the loaded CSV (even after case/whitespace-insensitive "
                f"matching). Available columns: {list(df.columns)}. Update "
                f"DATASET_CONFIG['{name}']['columns'] in data.py to match "
                f"the actual column names.")
        df = df[resolved]

    return df.select_dtypes(include=[np.number])


def preprocess(df: pd.DataFrame,
               seq_len: int,
               train_ratio: float = 0.70,
               seed: int = 42
               ) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, np.ndarray, np.ndarray]:
    """
    Fit MinMaxScaler on the train split only, build overlapping stride-1
    windows, shuffle, 70/30 split (Section 5.2).

    Returns train_seq, test_seq (windowed, normalised), the fitted scaler,
    and train_scaled/test_scaled -- the normalised but un-windowed,
    continuous version, which some baselines (e.g. DoppelGANger) need since
    they chunk the series into non-overlapping blocks themselves.
    """
    values = df.values.astype(np.float32)
    n_rows = len(values)

    # split before scaling, to avoid leakage
    split_idx = int(n_rows * train_ratio)
    train_raw = values[:split_idx]
    test_raw  = values[split_idx:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw).astype(np.float32)
    test_scaled  = scaler.transform(test_raw).astype(np.float32)

    def make_windows(arr: np.ndarray) -> np.ndarray:
        seqs = [arr[i: i + seq_len] for i in range(len(arr) - seq_len)]
        return np.array(seqs, dtype=np.float32)

    train_seq = make_windows(train_scaled)
    test_seq  = make_windows(test_scaled)

    rng = np.random.default_rng(seed)
    train_seq = rng.permutation(train_seq)

    print(f"  Train sequences : {train_seq.shape}")
    print(f"  Test  sequences : {test_seq.shape}")
    return train_seq, test_seq, scaler, train_scaled, test_scaled
