# PriTiGAN

This repository contains the implementation of the paper "PriTiGAN: A Privacy-Preserving Framework for Synthetic Time-Series Data Generation."

## About

Generative models can produce realistic synthetic time-series data for applications including energy consumption, financial markets, and network telemetry. However, training such models on real-world sensitive data introduces privacy risks, particularly through membership inference attacks that may reveal whether specific individuals or records were included in the training set. Conventional approaches typically apply differential privacy uniformly throughout the model, which can substantially impair the temporal dependencies and distributional characteristics that synthetic time-series models are intended to preserve.

This project proposes a dual-noise injection strategy that selectively applies differential privacy to the embedding and discriminator networks using DP-SGD, while the generator inherits the corresponding privacy guarantee through differential privacy's post-processing property. We evaluate the proposed approach against TimeGAN, DoppelGANger, and DP-TimeGAN across three real-world datasets covering energy consumption, stock-market data, and MBA telecom records. We assess privacy using black-box and white-box membership inference attacks, and we evaluate utility using Jensen–Shannon divergence (JSD), Wasserstein distance, dynamic time warping (DTW), Pearson correlation coefficient (PCC), and downstream classification and regression performance across a range of privacy budgets, ε.

## What's in here

```
pritigan/
  model.py          model + DP training logic
  data.py           dataset download/preprocessing
evaluation/
  metrics.py        JSD, WD, DTW, PCC, downstream classification/regression, memorization diagnostic
mia/
  attacks.py        black-box and white-box membership inference attacks
configs/
  default.yaml
train.py
privacy_sweep.py     noise_multiplier -> epsilon lookup, without a full training run
```

## Setup

```bash
git clone <repo-url>
cd PriTiGAN
pip install -r requirements.txt
```

`tensorflow-privacy==0.9.0` only installs on Python 3.9–3.11.

## Datasets

We use the following datasets for our implementation:

- **Stock** — Google (GOOG) daily historical prices, originally from [Yahoo Finance](https://finance.yahoo.com/quote/GOOG/history?p=GOOG), preprocessed and distributed by the original TimeGAN authors: [jsyoon0823/TimeGAN/data/stock_data.csv](https://github.com/jsyoon0823/TimeGAN/blob/master/data/stock_data.csv). 

- **Energy** — [UCI Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) (id 374), preprocessed and distributed by the original TimeGAN authors: [jsyoon0823/TimeGAN/data/energy_data.csv](https://github.com/jsyoon0823/TimeGAN/blob/master/data/energy_data.csv). 

- **MBA** — Measuring Broadband America, collected by the [FCC](https://www.fcc.gov/general/measuring-broadband-america), preprocessed to the same reduced form used by the original DoppelGANger authors ([fjxmlzn/DoppelGANger](https://github.com/fjxmlzn/DoppelGANger/blob/master/data/README.md)).


## Training

```bash
python train.py --dataset stock
```

10,000 iterations across autoencoder pretraining, supervisor pretraining, and joint DP training, targeting eps ~= 1. Override noise/clipping directly:

```bash
python train.py --dataset stock --noise_mult 1.5 --l2_clip 1.0
```

`--baseline` for ablated variants:

```bash
python train.py --dataset stock --baseline dptimegan       # discriminator-only DP
python train.py --dataset stock --baseline embedding_only  # embedding-only DP
python train.py --dataset stock --baseline nondp           # no DP
```

Each run saves loss curves, model weights, generated samples, and `privacy_accounting.json` (sampling rate, noise multiplier, clip norm, per-network update counts, epsilon) to `outputs/<dataset>/`.

## Evaluation

```bash
python evaluate.py --dataset stock
```

Fidelity metrics (JSD/WD/DTW/PCC), memorization check, both MIA attacks, PCA/t-SNE plots. Run classification and regression 5 times with different seeds, and report mean ± std.

## Memorization check

For each synthetic sequence, find its nearest real neighbor in the training set and in the test set. A memorizing generator will sit closer to training records than test records; a generalizing one should sit roughly equidistant from both. The "too close" threshold is calibrated against how close held-out test sequences naturally get to training sequences, rather than a fixed value.

Runs automatically as part of `evaluate.py`; shows up in the summary as a train/test distance ratio and a "% suspiciously close" figure.

## Privacy accounting

The embedding and discriminator networks are each modeled as a Poisson-subsampled Gaussian mechanism and composed via Rényi Differential Privacy (RDP), yielding a formal (ε, δ)-DP guarantee. The generator and recovery networks consume no additional privacy budget: as they operate exclusively on the outputs of the already-privatized embedding network, their privacy guarantee follows directly from the post-processing property of differential privacy.

To check what epsilon a noise multiplier gets before a full run:

```bash
python privacy_sweep.py --dataset stock --noise_mults 1.0 5.0 20.0 90.0
```

## MIA

- **Black-box (Monte Carlo)**, Hilprecht et al. — attacker queries the generator only. PCA to 5 components, then nearest-neighbor distance.
- **White-box (discriminator-based)**, Hayes et al. (LOGAN) — attacker has the discriminator and uses its confidence scores.

Both are reported as AUC; 0.5 = random guessing.

## Reproducibility

Seeded throughout (`seed=42` default, NumPy and TF). Original experiments ran on an Intel Xeon W-2255 with an RTX A5000 (24GB), CUDA 12.0.

## Citation

```
Padariya, D.; Taherkhani, A.; Boiten, E.; Wagner, I. "PriTiGAN: A Privacy-Preserving Framework for Synthetic Time-Series Data Generation." 
```

## License

MIT — see `LICENSE`.
