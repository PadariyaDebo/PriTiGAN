# PriTiGAN

Privacy-preserving GAN for synthetic time-series generation, built on TimeGAN. Instead of applying DP-SGD uniformly, noise is injected selectively into the embedding and discriminator networks; the generator inherits its privacy guarantee through post-processing.



## Layout

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
evaluate.py
```

## Setup

```bash
git clone <repo-url>
cd PriTiGAN
pip install -r requirements.txt
```

`tensorflow-privacy==0.9.0` only installs on Python 3.9–3.11 (silently unavailable on 3.12+) -- pin the Python version if setting up a fresh env.

## Datasets

Sourced from the same repos as the two baseline architectures this work builds on and compares against (TimeGAN [10], DoppelGANger [11]):

- **Stock** — Google (GOOG) daily historical prices (Open, High, Low, Close, Adj Close, Volume), originally from [Yahoo Finance](https://finance.yahoo.com/quote/GOOG/history?p=GOOG), preprocessed and distributed by the original TimeGAN authors: [jsyoon0823/TimeGAN/data/stock_data.csv](https://github.com/jsyoon0823/TimeGAN/blob/master/data/stock_data.csv). 

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

Fidelity metrics (JSD/WD/DTW/PCC), memorization check, both MIA attacks, PCA/t-SNE plots. Classification and regression each run 5 times with different seeds, reported as mean ± std.

## Memorization check

For each synthetic sequence, finds its nearest real neighbor in the training set and in the test set. A memorizing generator will sit closer to training records than test records; a generalizing one should sit roughly equidistant from both. The "too close" threshold is calibrated against how close held-out test sequences naturally get to training sequences, rather than a fixed value.

Runs automatically as part of `evaluate.py`; shows up in the summary as a train/test distance ratio and a "% suspiciously close" figure.

## Privacy accounting

Embedding and discriminator networks are each modeled as a Poisson-subsampled Gaussian mechanism, composed via Rényi DP, then converted to (ε, δ). Generator and recovery consume no additional budget -- they only touch DP-protected outputs, so their guarantee comes from post-processing.

To check what epsilon a noise multiplier gets before a full run:

```bash
python privacy_sweep.py --dataset stock --noise_mults 1.0 5.0 20.0 90.0
```

## MIA

- **Black-box (Monte Carlo)**, Hilprecht et al. — attacker queries the generator only. PCA to 5 components, then nearest-neighbor distance.
- **White-box (discriminator-based)**, Hayes et al. (LOGAN) — attacker has the discriminator and uses its confidence scores.

Both reported as AUC, 0.5 = random guessing.

## Reproducibility

Seeded throughout (`seed=42` default, numpy and TF). Original experiments ran on an Intel Xeon W-2255 with an RTX A5000 (24GB), CUDA 12.0.
