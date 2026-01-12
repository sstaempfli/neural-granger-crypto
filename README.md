# Neural Granger Crypto

This folder contains the notebooks and data pipeline for the neural Granger causality
experiments on crypto returns and volatility for BTC/ETH/SOL.

## Contents
- `preprocessing.ipynb`: fetches/raw downloads and builds processed feature files.
- `analysis.ipynb`: trains baselines and Neural/Static GVAR, runs backtests, and exports
  paper assets (tables/figures) into `paper_assets`.
- `data/`: raw and processed data (`data/processed/features_<SYMBOL>.parquet`).
- `checkpoints/`: saved model checkpoints for reuse.
- `paper_assets/`: exported CSV/LaTeX tables and PNG figures.

## Quick start
1) Install dependencies:
   `pip install -r requirements.txt` (pip, Python 3.11) or `conda env create -f environment.yml` (conda)
2) Run `preprocessing.ipynb` to create processed features.
3) Run `analysis.ipynb` to train/evaluate models and export assets.

## Notes
- Set `FAST_MODE = True` in `analysis.ipynb` for shorter debug runs.
- The analysis notebook expects `data/processed/features_<SYMBOL>.parquet` to exist.
