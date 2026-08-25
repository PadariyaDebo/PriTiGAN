"""
Data loading + preprocessing: sliding window stride=1, 70/30 split, seed=42.
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

DATASET_CONFIG = {
    "stock": {
        "seq_len":    24,
        "n_features": 6,
        "batch_size": 128,
        "columns":    ["Open", "High", "Low", "Close", "Adj Close", "Volume"],
    },
    "energy": {
        "seq_len":    24,
        "n_features": 28,
        "batch_size": 128,
        "columns":    ["Appliances", "lights", "T1", "RH_1", "T2", "RH_2",
                        "T3", "RH_3", "T4", "RH_4", "T5", "RH_5", "T6", "RH_6",
                        "T7", "RH_7", "T8", "RH_8", "T9", "RH_9", "T_out",
                        "Press_mm_hg", "RH_out", "Windspeed", "Visibility",
                        "Tdewpoint", "rv1", "rv2"],
    },
    "mba": {
        "seq_len":    56,
        "n_features": 2,
        "batch_size": 100,
        "columns":    ["traffic_byte_counter", "ping_loss_rate"],
    },
}


def load_dataset(name: str, data_path: str = None) -> pd.DataFrame:
    """Load a dataset from its remote URL, or from a local path"""
    if data_path is not None:
        df = pd.read_csv(data_path)
    else:
        if name not in DATA_URLS:
            raise ValueError(f"Unknown dataset '{name}'. "
                             f"Choose from {list(DATA_URLS.keys())} "
                             f"or supply data_path.")
        df = pd.read_csv(DATA_URLS[name])

    cfg = DATASET_CONFIG[name]

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
        raise ValueError(f"Dataset '{name}': columns {missing} not found. "
                         f"Available: {list(df.columns)}")

    return df[resolved].select_dtypes(include=[np.number])


def make_windows(arr: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Build overlapping sequences of length `seq_len` via a sliding window
    with stride 1.
    """
    n_windows = len(arr) - seq_len + 1
    if n_windows <= 0:
        raise ValueError(
            f"Array length ({len(arr)}) must be >= seq_len ({seq_len}) "
            f"to build at least one window."
        )
    seqs = [arr[i: i + seq_len] for i in range(n_windows)]
    return np.array(seqs, dtype=np.float32)


def preprocess(df: pd.DataFrame,
               seq_len: int,
               train_ratio: float = 0.70,
               seed: int = 42
               ) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, np.ndarray, np.ndarray]:
    """
    Split raw values into train/test (temporal order preserved for the
    split itself).
    Returns:
        train_seq   : windowed, shuffled training sequences
        test_seq    : windowed test sequences (order preserved)
        scaler      : the MinMaxScaler fitted on the training partition
        train_scaled: raw (unwindowed) scaled training values
        test_scaled : raw (unwindowed) scaled test values
    """
    values = df.values.astype(np.float32)
    n_rows = len(values)

    split_idx = int(n_rows * train_ratio)
    train_raw = values[:split_idx]
    test_raw  = values[split_idx:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw).astype(np.float32)
    test_scaled  = scaler.transform(test_raw).astype(np.float32)

    train_seq = make_windows(train_scaled, seq_len)
    test_seq  = make_windows(test_scaled, seq_len)

    # Shuffle only the training sequences (for SGD)
    rng = np.random.default_rng(seed)
    train_seq = rng.permutation(train_seq)

    print(f"  Train sequences : {train_seq.shape}")
    print(f"  Test  sequences : {test_seq.shape}")
    return train_seq, test_seq, scaler, train_scaled, test_scaled
    
