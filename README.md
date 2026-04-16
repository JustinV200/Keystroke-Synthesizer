# 🎯 Keystroke Synthesizer
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> **Transform text into realistic keystroke patterns using deep learning**

A neural network that learns individual typing behaviors and generates synthetic keystroke dynamics from text input. Built with a DeBERTa-v3 transformer and heteroscedastic regression for accurate timing prediction with calibrated uncertainty.

## 🌟 Features

- **🧠 Transformer Architecture**: DeBERTa-v3-base encoder with character-level expansion
- **⚡ Heteroscedastic Modeling**: Predicts both mean and variance for realistic per-keystroke variation
- **📊 Timing Prediction**: Dwell time, flight time, and typing speed
- **🛡️ Numerical Stability**: Gradient monitoring, NaN handling, and mixed-precision training
- **🔄 Real-time Synthesis**: Generate keystroke sequences from any text input
- **📈 Evaluation Tools**: Built-in accuracy testing and distribution visualization
- **🖥️ KeyForge Desktop App**: Tkinter UI that generates and replays keystrokes into any window

## 🏗️ Architecture

```
Text Input → DeBERTa Tokenizer → Transformer Encoder
                                        ↓
                              Token-to-Char Expansion
                                        ↓
                               Shared Backbone (768→256)
                                        ↓
                          ┌─────────────┴─────────────┐
                          ▼                           ▼
                    Mean Head                  LogVar Head
                   (3 features)               (3 features)
                          ↓                           ↓
                    [DwellTime,               [Uncertainty per
                     FlightTime,               feature, used for
                     typing_speed]             sampling at inference]
```

The model produces character-level predictions by expanding token embeddings to character positions via an offset mapping. Each character gets the embedding of the subword token that covers it, enabling per-character timing predictions.

## 📦 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/keystroke-synthesizer.git
cd keystroke-synthesizer

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

All training/testing code lives under `src/`. From the repo root, add it to `PYTHONPATH`:

```bash
export PYTHONPATH=src           # Linux / macOS / WSL
# or on Windows PowerShell:
$env:PYTHONPATH = "src"
```

### Training a Model
```python
from Trainer.Trainer import Trainer

trainer = Trainer()
trainer.train()
```

Or from the command line (run from the repo root with `PYTHONPATH=src`):
```bash
python -m Trainer.Trainer
```

### Generating Keystrokes
```python
from Testing.synthesizeKeystrokes import predict_keystrokes

predict_keystrokes(
    text_path="example/sample.txt",
    checkpoint_path="checkpoints/best_model.pt",
    output_csv="predicted_keystrokes.csv"
)
```

### SLURM Submission
A ready-to-go batch script is provided:
```bash
sbatch src/scripts/submit_jobs
```
It self-locates to the repo root, sets `PYTHONPATH=src`, and writes logs to `logs/job-<jobid>.{out,err}`.

### Desktop App (KeyForge)
For an interactive UI — generate, download, or live-replay keystrokes into any window:
```bash
cd KeyForge
python app.py
```
Requires a trained checkpoint and stats file at `KeyForge/Model/best_model.pt` and `KeyForge/Model/cont_stats.json`. See [`KeyForge/README.md`](KeyForge/README.md) for details.

### Evaluating Accuracy
```python
from Testing.accuracyTester import compare

compare()  # Compares original vs synthetic keystroke distributions
```

## 📊 Data Pipeline

1. **Preprocessing** (`src/dataPipeline/dataPrepper.py`)
   - Cleans raw keystroke CSV data (removes duplicates, invalid entries)
   - Replays edit sequences to keep only surviving keystrokes (backspace handling)
   - Computes DwellTime (key press duration), FlightTime (time between keys), and typing speed (CPM)
   - NaN-aware: FlightTime is NaN at non-consecutive transitions, typing speed NaN for first rows in rolling window

2. **Loading** (`src/dataPipeline/dataLoader.py`)
   - Pairs text files with keystroke CSVs by matching filenames
   - Standardizes continuous features (z-score) with persisted stats (`cont_stats.json`)
   - Builds character-to-token offset mapping for character-level predictions
   - Returns variable-length sequences (no target padding)

3. **Training** (`src/Trainer/Trainer.py`)
   - Heteroscedastic loss: Gaussian NLL + KL divergence penalty (annealed)
   - Mixed-precision training with gradient scaling
   - Gradient clipping and NaN/Inf monitoring
   - Cosine annealing warm restarts scheduler
   - Early stopping with patience

## 🎛️ Configuration

Training parameters in [`src/Trainer/config.py`](src/Trainer/config.py):

```python
BASE_MODEL   = "microsoft/deberta-v3-base"
MAX_TOKENS   = 512
EPOCHS       = 12
BATCH_SIZE   = 8
LR           = 1e-5
WEIGHT_DECAY = 0.01
PATIENCE     = 3

# KL annealing: gradually shift focus from mean accuracy to variance calibration
KL_WEIGHT_START  = 0.001
KL_WEIGHT_END    = 0.01
KL_ANNEAL_EPOCHS = 8

# Per-feature KL weights [DwellTime, FlightTime, typing_speed]
KL_FEATURE_WEIGHTS = [1.0, 0, 0.3]
```

## 📈 Results & Metrics

The model tracks:

- **Mean Absolute Error (MAE)**: Timing prediction accuracy
- **NLL Loss**: Gaussian negative log-likelihood (mean + variance fit)
- **KL Divergence**: Variance calibration toward empirical variance
- **Empirical Variance**: Computed from training data for regularization targets

## 🗂️ Project Structure

```
keystroke-synthesizer/
├── data/                       # Training data
│   ├── csv/                    # Keystroke timing CSVs
│   ├── texts/                  # Corresponding text files
│   └── cont_stats.json         # Standardization stats (written by loader)
├── src/                        # Training pipeline (add to PYTHONPATH)
│   ├── dataPipeline/           # Data processing
│   │   ├── dataPrepper.py      # CSV cleaning, edit replay, feature extraction
│   │   └── dataLoader.py       # Dataset class, standardization, tokenization
│   ├── Trainer/                # Training components
│   │   ├── Trainer.py          # Main training loop
│   │   ├── TextToKeystrokeModelMultiHead.py  # DeBERTa + regression heads
│   │   ├── HeteroscedasticKLLoss.py          # NLL + KL loss
│   │   ├── config.py           # Hyperparameters
│   │   ├── make_collate.py     # Variable-length batch collation
│   │   └── utils.py            # Empirical variance, NaN checks
│   ├── Testing/                # Evaluation
│   │   ├── synthesizeKeystrokes.py  # Inference: text file → keystroke CSV
│   │   ├── accuracyTester.py        # Distribution comparison
│   │   └── grapher.py               # Visualization plots
│   └── scripts/
│       └── submit_jobs         # SLURM batch script
├── KeyForge/                   # Desktop app (self-contained)
│   ├── app.py                  # Tkinter UI
│   ├── config.py               # Paths, model + UI defaults
│   ├── Model/                  # best_model.pt, cont_stats.json
│   └── Synthesize/
│       ├── load_model.py       # Tokenizer + checkpoint loader
│       ├── synthesize.py       # predict_keystrokes(text) → DataFrame
│       └── TextToKeystrokeModelMultiHead.py
├── example/                    # sample.txt + reference predicted_keystrokes.csv
├── checkpoints/                # Saved model weights
├── graphs/                     # Output plots
├── runs/                       # TensorBoard logs
└── logs/                       # SLURM job stdout/stderr
```

## 🔬 Technical Details

### Heteroscedastic Regression
The model predicts both mean and log-variance for each continuous feature at every character position. At inference, keystrokes are sampled from $\mathcal{N}(\mu, \sigma^2)$ where $\sigma^2 = e^{\text{logvar}} \cdot \text{std}_{\text{train}}^2$, producing realistic variation rather than deterministic outputs.

### KL Annealing
KL weight increases linearly from 0.001 to 0.01 over the first 8 epochs. This lets the model focus on learning accurate means first, then gradually calibrate variance predictions toward the empirical variance of the training data.

### Character-Level Expansion
DeBERTa produces subword token embeddings, but keystroke timing is per-character. The model uses offset mappings to expand token embeddings to character positions via `torch.gather`, giving each character the embedding of its covering token.

### Edit Replay
Raw keystroke logs include backspaces and non-producing keys. The data pipeline replays the edit sequence to identify which keystrokes actually survived in the final text, then recomputes FlightTime only between consecutive survivors.

## 📊 Dataset

**Source**: [KLiCKe Dataset](https://www.kaggle.com/datasets/julesking/tla-lab-pii-competition-dataset?resource=download-directory)

The dataset contains text-keystroke pairs with detailed timing information:
- **Dwell Time**: Key press duration (capped at 300ms)
- **Flight Time**: Time between consecutive keystrokes (capped at 900ms)
- **Typing Speed**: Characters per minute over a rolling window (capped at 490 CPM)

## 🙏 Acknowledgments

- **Dataset**: [KLiCKe Competition Dataset](https://www.kaggle.com/datasets/julesking/tla-lab-pii-competition-dataset?resource=download-directory)
- **Model Architecture**: Microsoft DeBERTa-v3
- **Framework**: PyTorch & Hugging Face Transformers

---

<div align="center">

## Disclaimer:
    Credit to claude for making this readme look a lot nicer then I could 😊

</div>
