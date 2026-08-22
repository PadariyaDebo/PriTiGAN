"""
Membership inference attacks: black-box Monte Carlo
(Hilprecht et al., generator-only access, PCA + nearest-distance scoring)
and white-box discriminator-based (Hayes et al. / LOGAN, uses discriminator
confidence directly). Both reported as AUC, 5 runs, mean ± std, 0.5 = random.
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from typing import Tuple, Dict


# ── Black-box: Monte Carlo MIA ────────────────────────────────────────────────
def monte_carlo_mia(generator_fn,
                    member_seqs:     np.ndarray,
                    non_member_seqs: np.ndarray,
                    n_mc_samples:    int = 1000,
                    n_pca_components: int = 5,
                    seed:            int = 42) -> float:
    """
    Black-box Monte Carlo MIA (Hilprecht et al., 2019).

    generator_fn : (n,) -> (n, seq_len, n_features), the trained generator
    member_seqs / non_member_seqs : training / holdout samples
    n_mc_samples : number of synthetic samples to draw
    n_pca_components : PCA components (paper uses 5)

    Returns AUC (0.50 = random guess).
    """
    np.random.seed(seed)

    # Generate synthetic samples
    synth = generator_fn(n_mc_samples)                  # (n_mc, seq, feat)

    # Flatten sequences for PCA
    m_2d  = member_seqs.reshape(len(member_seqs),      -1)
    nm_2d = non_member_seqs.reshape(len(non_member_seqs), -1)
    s_2d  = synth.reshape(n_mc_samples,                 -1)

    # Fit PCA jointly on all data (member + non-member + synthetic)
    all_data = np.concatenate([m_2d, nm_2d, s_2d], axis=0)
    pca = PCA(n_components=n_pca_components, random_state=seed)
    pca.fit(all_data)

    m_proj  = pca.transform(m_2d)
    nm_proj = pca.transform(nm_2d)
    s_proj  = pca.transform(s_2d)

    # Membership score: negative min Euclidean distance to synthetic samples
    # (closer to synthetic → more likely training member)
    def min_dist_score(candidate_proj):
        scores = []
        for c in candidate_proj:
            dists = np.linalg.norm(s_proj - c, axis=1)
            scores.append(-dists.min())          # negate: higher = closer
        return np.array(scores)

    scores_m  = min_dist_score(m_proj)
    scores_nm = min_dist_score(nm_proj)

    y_true = np.concatenate([np.ones(len(member_seqs)),
                              np.zeros(len(non_member_seqs))])
    y_score = np.concatenate([scores_m, scores_nm])

    return float(roc_auc_score(y_true, y_score))


# ── White-box: Discriminator-based MIA ───────────────────────────────────────
def discriminator_mia(embedder_fn,
                      discriminator_fn,
                      member_seqs:     np.ndarray,
                      non_member_seqs: np.ndarray) -> float:
    """
    White-box discriminator MIA (Hayes et al. LOGAN, 2019) — Section 6.1.2.
    Discriminator confidence (P(real)) is used directly as membership score.

    embedder_fn      : (x) -> latent H
    discriminator_fn : (H) -> confidence in [0,1]
    member_seqs / non_member_seqs : training / holdout samples

    Returns AUC (0.50 = random guess).
    """
    def get_scores(seqs):
        h      = embedder_fn(seqs)
        scores = discriminator_fn(h)
        # Average across time steps → scalar per sequence
        return scores.numpy().mean(axis=(1, 2))

    scores_m  = get_scores(member_seqs.astype(np.float32))
    scores_nm = get_scores(non_member_seqs.astype(np.float32))

    y_true  = np.concatenate([np.ones(len(member_seqs)),
                               np.zeros(len(non_member_seqs))])
    y_score = np.concatenate([scores_m, scores_nm])

    return float(roc_auc_score(y_true, y_score))


# ── Run full MIA evaluation (5 runs, mean ± std) ─────────────────────────────
def evaluate_mia(model,
                 train_seq:   np.ndarray,
                 test_seq:    np.ndarray,
                 n_runs:      int = 5,
                 member_size: int = 500,
                 seed:        int = 42) -> Dict[str, float]:
    """
    Both attacks over n_runs independent trials, mean ± std (Section 6.1.3).

    model       : trained PriTiGAN instance
    train_seq   : samples for the member set
    test_seq    : samples for the non-member set
    member_size : samples per class (paper uses 500)

    Returns dict: bb_auc_mean, bb_auc_std, wb_auc_mean, wb_auc_std
    """
    bb_aucs, wb_aucs = [], []

    for run in range(n_runs):
        rng = np.random.default_rng(seed + run)

        # Sample member (training) and non-member (test) sets
        n_m  = min(member_size, len(train_seq))
        n_nm = min(member_size, len(test_seq))

        m_idx  = rng.permutation(len(train_seq))[:n_m]
        nm_idx = rng.permutation(len(test_seq))[:n_nm]

        members     = train_seq[m_idx].astype(np.float32)
        non_members = test_seq[nm_idx].astype(np.float32)

        # ── Black-box MIA ─────────────────────────────────────────────────
        bb_auc = monte_carlo_mia(
            generator_fn=model.generate,
            member_seqs=members,
            non_member_seqs=non_members,
            n_mc_samples=1000,
            n_pca_components=5,
            seed=seed + run,
        )
        bb_aucs.append(bb_auc)

        # ── White-box MIA ─────────────────────────────────────────────────
        wb_auc = discriminator_mia(
            embedder_fn=lambda x: model.embedder(x, training=False),
            discriminator_fn=lambda h: model.discriminator(h, training=False),
            member_seqs=members,
            non_member_seqs=non_members,
        )
        wb_aucs.append(wb_auc)

        print(f"  Run {run + 1}/{n_runs} | BB-AUC: {bb_auc:.3f} | "
              f"WB-AUC: {wb_auc:.3f}")

    results = {
        "bb_auc_mean": float(np.mean(bb_aucs)),
        "bb_auc_std":  float(np.std(bb_aucs)),
        "wb_auc_mean": float(np.mean(wb_aucs)),
        "wb_auc_std":  float(np.std(wb_aucs)),
    }

    print(f"\n  BB AUC: {results['bb_auc_mean']:.3f} ± {results['bb_auc_std']:.3f}")
    print(f"  WB AUC: {results['wb_auc_mean']:.3f} ± {results['wb_auc_std']:.3f}")
    print(f"  (Random-guess baseline: 0.500)")
    return results
