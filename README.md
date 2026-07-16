# silicon-analysis-lstm

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Domain](https://img.shields.io/badge/Domain-Environmental%20ML-2e8b57?style=flat)

## Overview

This project implements a Long Short-Term Memory (LSTM) neural network for time-series forecasting of silicon (Si) concentration in rivers across multiple monitoring sites. Dissolved silicon is a critical indicator of watershed health, biogeochemical cycling, and aquatic ecosystem dynamics, yet it is difficult to measure at scale. By learning spatiotemporal patterns from hydro-chemical and static site features, this model achieves a **22% improvement in forecasting accuracy** over baseline approaches — enabling more reliable, data-driven water quality assessment without continuous physical sampling.

## Features

- Multi-site forecasting across 9 stream monitoring locations
- Sliding-window sequence modeling (window length = 10 timesteps)
- Separate MinMaxScaler for the silicon target variable to prevent data leakage
- Per-stream 80/20 train/test splitting for robust generalization
- CUDA-accelerated training with automatic GPU utilization
- Model and scaler artifacts persisted to disk for reproducible inference
- End-to-end pipeline: raw CSV ingestion → preprocessing → training → evaluation plots

## Tech Stack

- **Language:** Python
- **Deep Learning:** PyTorch (`torch.nn.LSTM`, CUDA)
- **Data Processing:** pandas, NumPy
- **Feature Scaling:** scikit-learn (`MinMaxScaler`)
- **Visualization:** matplotlib
- **Model Architecture:** 2-layer LSTM (hidden size 1024, input size 29) with a fully-connected output head

## Model Architecture

| Parameter | Value |
|---|---|
| Model type | LSTM |
| Input size | 29 features |
| Hidden size | 1024 |
| Number of layers | 2 |
| Sequence length | 10 |
| Training epochs | 20 |
| Loss function | MSELoss |
| Optimizer | Adam |

## Getting Started

### Prerequisites

Python 3.8 or higher and a CUDA-capable GPU are recommended.

### Installation

1. Clone the repository:

```bash
git clone https://github.com/katkhedepushpak/silicon-analysis-lstm.git
cd silicon-analysis-lstm
```

2. Install dependencies:

```bash
pip install torch numpy pandas scikit-learn matplotlib
```

### Data

Place the following input files in the project root before running:

- `master_9_sites.csv` — Time-series hydro-chemical measurements merged by stream site
- `static.csv` — Static site-level attributes merged on the `Stream` column

### Usage

Run the full pipeline (preprocessing → training → evaluation):

```bash
python main.py
```

This will:
1. Load and merge `master_9_sites.csv` and `static.csv`
2. Scale features and create sliding-window sequences per stream
3. Train the LSTM model for 20 epochs on GPU (if available)
4. Save the trained model weights and scaler artifacts to disk
5. Generate and display result plots

To modify training parameters, edit the constants at the top of `main.py` (epochs, hidden size, etc.) before running.

### Inference

After training, the saved model state dict and MinMaxScaler artifacts can be loaded for inference on new stream data without retraining.

## Project Structure

```
silicon-analysis-lstm/
├── main.py            # Pipeline orchestration
├── model.py           # LSTMModel definition (nn.Module)
├── preprocessing.py   # Data loading, scaling, sequence creation
├── train.py           # Training loop (MSELoss + Adam)
├── master_9_sites.csv # (required) Time-series site measurements
└── static.csv         # (required) Static site attributes
```

## Author

Built by [Pushpak Vijay Katkhede](https://katkhedepushpak.github.io) — Software Engineer at Oregon State University, MS Computer Science (GPA 3.75), with 3+ years of backend and cloud engineering experience at IBM. Published at ICSE 2026 (ACM/IEEE).
