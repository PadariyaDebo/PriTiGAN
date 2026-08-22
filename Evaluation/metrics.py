"""
Evaluation metrics: JSD, WD, DTW, PCC for fidelity;
GRU/RNN-based classification + regression for downstream TSTR; PCA/t-SNE
for visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, SimpleRNN, Dense
from tensorflow.keras.callbacks import EarlyStopping
from typing import Dict, List, Tuple


# ── 1. Jensen-Shannon Divergence ─────────────────────────────────────────────
def compute_jsd(real: np.ndarray, synth: np.ndarray,
                n_bins: int = 30) -> Dict[str, float]:
    
    real_2d  = real.reshape(-1,  real.shape[-1])
    synth_2d = synth.reshape(-1, synth.shape[-1])
    n_feat   = real_2d.shape[1]
    results  = {}
    for i in range(n_feat):
        r_f = real_2d[:, i].flatten()
        s_f = synth_2d[:, i].flatten()
        r_hist, bins = np.histogram(r_f, bins=n_bins, density=True)
        s_hist, _    = np.histogram(s_f, bins=bins,   density=True)
        r_hist = r_hist / np.sum(r_hist)
        s_hist = s_hist / np.sum(s_hist)
        results[i] = float(jensenshannon(r_hist, s_hist))
    return results


# ── 2. Wasserstein Distance ───────────────────────────────────────────────────
def compute_wd(real: np.ndarray, synth: np.ndarray) -> Dict[str, float]:
  
    real_2d  = real.reshape(-1,  real.shape[-1])
    synth_2d = synth.reshape(-1, synth.shape[-1])
    n_feat   = real_2d.shape[1]
    results  = {}
    for i in range(n_feat):
        results[i] = float(wasserstein_distance(real_2d[:, i], synth_2d[:, i]))
    return results


# ── 3. Dynamic Time Warping ───────────────────────────────────────────────────
def compute_dtw(real: np.ndarray, synth: np.ndarray,
                sample_size: int = 500) -> float:
    
    try:
        from dtaidistance import dtw_ndim
    except ImportError:
        raise ImportError("Install dtaidistance: pip install dtaidistance")

    n = min(sample_size, len(real), len(synth))
    total = sum(
        dtw_ndim.distance(
            np.ascontiguousarray(real[i],  dtype=np.double),
            np.ascontiguousarray(synth[i], dtype=np.double))
        for i in range(n)
    )
    return total / n


# ── 4. Pearson Correlation Coefficient ───────────────────────────────────────
def compute_pcc(data: np.ndarray) -> Dict[int, float]:
   
    data_2d = data.reshape(-1, data.shape[-1])
    n_feat  = data_2d.shape[1]

    if n_feat < 2:
        return {0: 1.0} if n_feat == 1 else {}

    corr_matrix = np.corrcoef(data_2d, rowvar=False)  # (n_feat, n_feat)
    results = {}
    for i in range(n_feat):
        others = [corr_matrix[i, j] for j in range(n_feat) if j != i]
        results[i] = float(np.mean(others))
    return results


# ── 5. Classification (TSTR) ─────────────────────────────────────────────────
def evaluate_classification(train_synth: np.ndarray,
                             test_real:   np.ndarray,
                             epochs:      int = 250,
                             batch_size:  int = 128,
                             seed:        int = 42) -> float:
   
    tf.random.set_seed(seed)
    seq_len, n_feat = train_synth.shape[1], train_synth.shape[2]

    # Build binary labels: last-step direction
    def make_labels(data):
        return (data[:, -1, 0] > data[:, -2, 0]).astype(np.float32)

    y_train = make_labels(train_synth)
    y_test  = make_labels(test_real)

    model = Sequential([
        GRU(units=24, input_shape=(seq_len, n_feat), name="GRU"),
        Dense(1, activation="sigmoid", name="OUT"),
    ], name="GRU_Classifier")
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    model.fit(train_synth, y_train,
              epochs=epochs,
              batch_size=batch_size,
              verbose=0,
              callbacks=[EarlyStopping(patience=20, restore_best_weights=True)])

    _, acc = model.evaluate(test_real, y_test, verbose=0)
    return float(acc)


# ── 6. Regression (TSTR) ─────────────────────────────────────────────────────
def evaluate_regression(train_synth: np.ndarray,
                         test_real:   np.ndarray,
                         epochs:      int = 250,
                         batch_size:  int = 128,
                         seed:        int = 42) -> Tuple[float, float]:
    
    tf.random.set_seed(seed)
    seq_len, n_feat = train_synth.shape[1], train_synth.shape[2]

    X_train = train_synth[:, :-1, :]
    y_train = train_synth[:, -1, :]
    X_test  = test_real[:,  :-1, :]
    y_test  = test_real[:,  -1, :]

    model = Sequential([
        SimpleRNN(50, input_shape=(seq_len - 1, n_feat), activation="relu"),
        Dense(n_feat, activation="sigmoid"),
    ], name="RNN_Regressor")
    model.compile(optimizer="adam", loss="mean_absolute_error")
    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              epochs=epochs,
              batch_size=batch_size,
              verbose=0,
              callbacks=[EarlyStopping(monitor="val_loss", patience=20,
                                       restore_best_weights=True)])

    y_pred = model.predict(X_test, verbose=0)
    r2  = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    return r2, mae


# ── 7. PCA and t-SNE visualisation ───────────────────────────────────────────
def plot_pca_tsne(real: np.ndarray, synth: np.ndarray,
                  sample_size: int = 250,
                  save_path: str = None,
                  seed: int = 42):
    
    np.random.seed(seed)
    n = min(sample_size, len(real), len(synth))
    idx_r = np.random.permutation(len(real))[:n]
    idx_s = np.random.permutation(len(synth))[:n]

    seq_len = real.shape[1]
    r_2d = real[idx_r].reshape(-1, seq_len)
    s_2d = synth[idx_s].reshape(-1, seq_len)

    # PCA — fit on real only, matching the notebook
    pca = PCA(n_components=2, random_state=seed)
    pca.fit(r_2d)
    r_pca = pca.transform(r_2d)
    s_pca = pca.transform(s_2d)

    # t-SNE — fit jointly on real+synthetic
    combined = np.concatenate([r_2d, s_2d], axis=0)
    tsne_res = TSNE(n_components=2, random_state=seed,
                    perplexity=40, max_iter=300).fit_transform(combined)
    r_tsne = tsne_res[:len(r_2d)]
    s_tsne = tsne_res[len(r_2d):]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, r, s, title in zip(
            axes,
            [r_pca,  r_tsne],
            [s_pca,  s_tsne],
            ["PCA results", "t-SNE results"]):
        ax.scatter(r[:, 0], r[:, 1], c="black", alpha=0.3, label="real",      s=10)
        ax.scatter(s[:, 0], s[:, 1], c="red",   alpha=0.3, label="synthetic", s=10)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved PCA/t-SNE plot → {save_path}")
    plt.show()


# ── 8. Memorization / generalization diagnostic ──────────────────────────────
def evaluate_memorization(synth: np.ndarray,
                          train_seq: np.ndarray,
                          test_seq:  np.ndarray,
                          n_pca_components: int = 5,
                          sample_size: int = 500,
                          seed: int = 42) -> Dict[str, float]:
    
    rng = np.random.default_rng(seed)
    n = min(sample_size, len(synth), len(train_seq), len(test_seq))

    synth_s = synth[rng.permutation(len(synth))[:n]]
    train_s = train_seq[rng.permutation(len(train_seq))[:n]]
    test_s  = test_seq[rng.permutation(len(test_seq))[:n]]

    s_2d  = synth_s.reshape(n, -1)
    tr_2d = train_s.reshape(n, -1)
    te_2d = test_s.reshape(n, -1)

    pca = PCA(n_components=n_pca_components, random_state=seed)
    pca.fit(np.concatenate([tr_2d, te_2d, s_2d], axis=0))
    s_proj  = pca.transform(s_2d)
    tr_proj = pca.transform(tr_2d)
    te_proj = pca.transform(te_2d)

    def nearest_dists(query_proj, reference_proj):
        dists = np.empty(len(query_proj))
        for i, q in enumerate(query_proj):
            dists[i] = np.linalg.norm(reference_proj - q, axis=1).min()
        return dists

    dist_to_train = nearest_dists(s_proj, tr_proj)
    dist_to_test  = nearest_dists(s_proj, te_proj)

    real_nn_dists = nearest_dists(te_proj, tr_proj)
    suspicious_threshold = 0.1 * np.median(real_nn_dists)
    frac_suspicious = float(np.mean(dist_to_train < suspicious_threshold))

    mean_train = float(np.mean(dist_to_train))
    mean_test  = float(np.mean(dist_to_test))

    return {
        "mean_dist_to_train": mean_train,
        "mean_dist_to_test":  mean_test,
        "train_test_ratio":   mean_train / mean_test if mean_test > 0 else float('nan'),
        "frac_suspiciously_close": frac_suspicious,
    }


def evaluate_all(real: np.ndarray,
                 synth: np.ndarray,
                 dataset_name: str = "",
                 run_downstream: bool = True,
                 n_downstream_runs: int = 5,
                 train_seq: np.ndarray = None,
                 seed: int = 42) -> dict:
    
    print(f"\n{'='*60}")
    print(f" Evaluation — {dataset_name}")
    print(f"{'='*60}")

    results = {}

    # JSD
    jsd = compute_jsd(real, synth)
    mean_jsd = np.mean(list(jsd.values()))
    results["jsd"] = jsd
    results["jsd_mean"] = mean_jsd
    print(f"\n  JSD  (mean): {mean_jsd:.4f}")
    for k, v in jsd.items():
        print(f"    Feature {k}: {v:.4f}")

    # WD
    wd = compute_wd(real, synth)
    mean_wd = np.mean(list(wd.values()))
    results["wd"] = wd
    results["wd_mean"] = mean_wd
    print(f"\n  WD   (mean): {mean_wd:.4f}")
    for k, v in wd.items():
        print(f"    Feature {k}: {v:.4f}")

    # DTW
    try:
        dtw_score = compute_dtw(real, synth)
        results["dtw"] = dtw_score
        print(f"\n  DTW  (mean): {dtw_score:.4f}")
    except ImportError:
        print("\n  DTW: skipped (install dtaidistance)")

    pcc_real  = compute_pcc(real)
    pcc_synth = compute_pcc(synth)
    results["pcc_real"]  = pcc_real
    results["pcc_synth"] = pcc_synth
    print(f"\n  PCC  (feature-wise, avg. correlation with other features):")
    print(f"    {'Feature':<10}{'Real':>10}{'Synthetic':>12}")
    for k in pcc_real:
        print(f"    {k:<10}{pcc_real[k]:>10.4f}{pcc_synth.get(k, float('nan')):>12.4f}")

    # only runs if the caller passed the real training split
    if train_seq is not None:
        mem = evaluate_memorization(synth, train_seq, real, seed=seed)
        results["memorization"] = mem
        print(f"\n  Memorization diagnostic (nearest-neighbor, PCA space):")
        print(f"    Mean dist to train : {mem['mean_dist_to_train']:.4f}")
        print(f"    Mean dist to test  : {mem['mean_dist_to_test']:.4f}")
        print(f"    Train/test ratio   : {mem['train_test_ratio']:.4f}  "
              f"(~1.0 = no memorization signal; << 1.0 = warning sign)")
        print(f"    Suspiciously close : {mem['frac_suspiciously_close']*100:.2f}%")

    # optional (slower) — n_downstream_runs trials, different seed each time
    if run_downstream:
        print(f"\n  Downstream tasks (TSTR) — {n_downstream_runs} runs ...")
        n_train = int(len(synth) * 0.7)
        train_s = synth[:n_train]
        test_r  = real

        accs, r2s, maes = [], [], []
        for run in range(n_downstream_runs):
            run_seed = seed + run
            acc = evaluate_classification(train_s, test_r, seed=run_seed)
            r2, mae = evaluate_regression(train_s, test_r, seed=run_seed)
            accs.append(acc)
            r2s.append(r2)
            maes.append(mae)
            print(f"    Run {run + 1}/{n_downstream_runs} | "
                  f"Acc: {acc*100:.2f}% | R²: {r2:.4f} | MAE: {mae:.4f}")

        results["classification_accuracy"]     = float(np.mean(accs))
        results["classification_accuracy_std"] = float(np.std(accs))
        results["r2"]      = float(np.mean(r2s))
        results["r2_std"]  = float(np.std(r2s))
        results["mae"]     = float(np.mean(maes))
        results["mae_std"] = float(np.std(maes))

        print(f"\n    Classification accuracy : "
              f"{results['classification_accuracy']*100:.2f}% "
              f"± {results['classification_accuracy_std']*100:.2f}%")
        print(f"    Regression R²           : "
              f"{results['r2']:.4f} ± {results['r2_std']:.4f}")
        print(f"    Regression MAE          : "
              f"{results['mae']:.4f} ± {results['mae_std']:.4f}")

    print(f"\n{'='*60}\n")
    return results
