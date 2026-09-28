"""
Evaluation: fidelity metrics, downstream TSTR tasks, black/white-box
MIA and PCA/t-SNE plots, run against a trained model's saved outputs.
"""

import argparse
import numpy as np
from pathlib import Path

from PriTi.model          import PriTiGAN
from PriTi.data           import DATASET_CONFIG
from Evaluation.metrics   import evaluate_all, plot_pca_tsne
from MIA.attacks          import evaluate_mia

def run_evaluation(dataset_name: str,
                   output_dir:   str = "outputs",
                   n_mia_runs:   int = 5,
                   seed:         int = 42,
                   baseline:     str = "pritigan"):

    cfg       = DATASET_CONFIG[dataset_name]
    out       = (Path(output_dir) / dataset_name / baseline
                if baseline != "pritigan" else Path(output_dir) / dataset_name)

    # Load saved sequences 
    print(f"Loading data from {out}/ ...")
    train_seq = np.load(out / "train_sequences.npy")
    test_seq  = np.load(out / "test_sequences.npy")
    synth     = np.load(out / "synthetic_sequences.npy")

    print(f"  Train:     {train_seq.shape}")
    print(f"  Test:      {test_seq.shape}")
    print(f"  Synthetic: {synth.shape}")

    #  Fidelity + Downstream metrics 
    n_eval = min(len(test_seq), len(synth))
    results = evaluate_all(
        real=test_seq[:n_eval],
        synth=synth[:n_eval],
        dataset_name=dataset_name,
        run_downstream=True,
        train_seq=train_seq,
        seed=seed,
    )

    #  PCA / t-SNE 
    plot_pca_tsne(
        real=test_seq,
        synth=synth,
        sample_size=250,
        save_path=str(out / "pca_tsne.png"),
        seed=seed,
    )

    #  MIA 
    print("\nLoading model weights for MIA evaluation...")
    model = PriTiGAN({
        "seq_len":          cfg["seq_len"],
        "n_features":       cfg["n_features"],
        "hidden_dim":       24,
        "num_layers":       3,
        "gamma":            1,
        "lambda1":          10,
        "lambda2":          0.1,
        "noise_multiplier": 1.0,   # value does not affect inference
        "l2_norm_clip":     1.0,
        "num_microbatches": 1,
        "learning_rate":    5e-4,
    })
    dummy = np.zeros((1, cfg["seq_len"], cfg["n_features"]), dtype=np.float32)
    _ = model.embedder(dummy)
    _ = model.discriminator(model.embedder(dummy))

    model.embedder.load_weights(str(out / "embedder.weights.h5"))
    model.discriminator.load_weights(str(out / "discriminator.weights.h5"))
    model.generator.load_weights(str(out / "generator.weights.h5"))
    model.recovery.load_weights(str(out / "recovery.weights.h5"))
    model.supervisor.load_weights(str(out / "supervisor.weights.h5"))

    print(f"\nMIA evaluation ({n_mia_runs} runs) ...")
    mia_results = evaluate_mia(
        model=model,
        train_seq=train_seq,
        test_seq=test_seq,
        n_runs=n_mia_runs,
        member_size=500,
        seed=seed,
    )
    results.update(mia_results)

    # Summary 
    print("\n" + "="*50)
    print(" SUMMARY")
    print("="*50)
    print(f"  JSD mean          : {results.get('jsd_mean', 'N/A'):.4f}")
    print(f"  WD  mean          : {results.get('wd_mean',  'N/A'):.4f}")
    print(f"  DTW mean          : {results.get('dtw',      'N/A'):.4f}")
    print(f"  Classification Acc: {results.get('classification_accuracy', 0)*100:.2f}% "
          f"± {results.get('classification_accuracy_std', 0)*100:.2f}%")
    print(f"  Regression R²     : {results.get('r2', float('nan')):.4f} "
          f"± {results.get('r2_std', 0):.4f}")
    print(f"  Regression MAE    : {results.get('mae', float('nan')):.4f} "
          f"± {results.get('mae_std', 0):.4f}")
    print(f"  BB-AUC (MIA)      : {results['bb_auc_mean']:.3f} ± {results['bb_auc_std']:.3f}")
    if "memorization" in results:
        print(f"  Memorization ratio: {results['memorization']['train_test_ratio']:.4f} "
              f"({results['memorization']['frac_suspiciously_close']*100:.1f}% suspiciously close)")
    print(f"  WB-AUC (MIA)      : {results['wb_auc_mean']:.3f} ± {results['wb_auc_std']:.3f}")
    print("="*50)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained PriTiGAN model")
    parser.add_argument("--dataset",    type=str, default="stock",
                        choices=["stock", "energy", "mba"])
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--n_mia_runs", type=int, default=5)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--baseline",   type=str, default="pritigan",
                        choices=["pritigan", "dptimegan", "embedding_only", "nondp"],
                        help="Which trained baseline/ablation variant to evaluate "
                             "(must match what was passed to train.py --baseline)")
    args = parser.parse_args()

    run_evaluation(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        n_mia_runs=args.n_mia_runs,
        seed=args.seed,
        baseline=args.baseline,
    )
