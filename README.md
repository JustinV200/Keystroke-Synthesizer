# 🎯 Keystroke Synthesizer
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> **Transform text into realistic keystroke patterns using deep learning**

A  neural network that learns individual typing behaviors and generates synthetic keystroke dynamics from text input. Built with transformer architecture and heteroscedastic regression for accurate timing prediction.

## 🌟 Features

- **🧠 Advanced Architecture**: DeBERTa-v3 transformer with multi-head prediction
- **⚡ Heteroscedastic Modeling**: Predicts both mean and uncertainty for realistic variation  
- **📊 Comprehensive Metrics**: Dwell time, flight time, typing speed, and keystroke flags
- **🛡️ Numerical Stability**: Conservative bounds and gradient monitoring for robust training
- **🔄 Real-time Synthesis**: Generate keystroke sequences from any text input
- **📈 Performance Tracking**: Built-in accuracy testing and visualization tools

## 🏗️ Architecture

```
Text Input → DeBERTa Tokenizer → Transformer Encoder
                                        ↓
                               Shared Backbone (512→256)
                                        ↓
                          ┌─────────────┼─────────────┐
                          ▼             ▼             ▼
                    Mean Head    LogVar Head   Classification Head
                   (3 features)  (3 features)    (7 binary flags)
                          ↓             ↓             ↓
                    [DwellTime,   [Uncertainty]   [is_letter,
                     FlightTime,                   is_digit, ...]
                     typing_speed]
```

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

# Download and prepare dataset (see Data section)
```

## 🚀 Quick Start

### Training a Model
```python
from Trainer.Trainer import Trainer

# Initialize and train
trainer = Trainer()
trainer.train()
```

### Generating Keystrokes
```python
from Testing.synthesizeKeystrokes import predict_keystrokes

# Generate keystroke pattern for text
text_path = "sample.txt"
predict_keystrokes(
    text_path=text_path,
    checkpoint_path="checkpoints/best_model.pt",
    output_csv="predicted_keystrokes.csv"
)
```

### Testing Accuracy
```python
from Testing.accuracyTester import compare

# Evaluate model performance
compare()  # Compares original vs synthetic keystroke statistics
```
```

## 📊 Data Pipeline

The system processes keystroke data through a robust pipeline:

1. **Data Preprocessing** (`dataPipeline/dataPrepper.py`)
   - Outlier detection and capping
   - Feature engineering and validation
   - Comprehensive NaN handling

2. **Data Loading** (`dataPipeline/dataLoader.py`) 
   - NaN-aware standardization
   - Statistics persistence via JSON
   - Efficient batch processing

3. **Model Training** (`Trainer/Trainer.py`)
   - Conservative numerical bounds
   - Enhanced gradient monitoring  
   - Early stopping with validation

## 🎛️ Configuration

Key training parameters in [`Trainer/config.py`](Trainer/config.py):

```python
# Model Configuration
BASE_MODEL = "microsoft/deberta-v3-base"
MAX_TOKENS = 512

# Training Configuration  
EPOCHS = 12
BATCH_SIZE = 8
LR = 1e-5  # Conservative for stability
WEIGHT_DECAY = 0.01

# KL Regularization
KL_WEIGHT_START = 0.001
KL_WEIGHT_END = 0.03
KL_ANNEAL_EPOCHS = 8

# Feature-specific weights [DwellTime, FlightTime, typing_speed]
KL_FEATURE_WEIGHTS = [1.0, 0, 0.3]
```

## 📈 Results & Metrics

The model tracks multiple performance indicators:

- **Mean Absolute Error (MAE)**: Timing prediction accuracy
- **KL Divergence**: Uncertainty calibration quality  
- **Classification Accuracy**: Keystroke type prediction
- **Empirical Variance**: Realistic variation modeling

## 🗂️ Project Structure

```
keystroke-synthesizer/
├── 📁 data/                    # Training data
│   ├── csv/                    # Keystroke timing data
│   ├── texts/                  # Corresponding text samples
│   └── predicted_csvs/         # Generated synthetic keystroke data
├── 📁 dataPipeline/           # Data processing pipeline
│   ├── __init__.py            # Package initialization
│   ├── dataPrepper.py         # Data cleaning & preprocessing
│   └── dataLoader.py          # Dataset loading & standardization
├── 📁 Trainer/                # Training components
│   ├── __init__.py            # Package initialization
│   ├── Trainer.py             # Main training class
│   ├── TextToKeystrokeModelMultiHead.py  # Model architecture
│   ├── HeteroscedasticKLLoss.py          # Loss function
│   ├── config.py              # Training configuration
│   ├── make_collate.py        # Batch processing
│   └── utils.py               # Training utilities
├── 📁 Testing/                # Evaluation and analysis tools
│   ├── synthesizeKeystrokes.py # Text-to-keystroke generation
│   ├── accuracyTester.py      # Model evaluation & comparison
│   └── grapher.py             # Results visualization
├── 📁 checkpoints/            # Saved models
├── 📁 graphs/                 # Performance visualizations
├── 📁 misc/                   # Miscellaneous utilities
├── 📁 runs/                   # Training logs and outputs
└── README.md                  # This file
```

## 🔬 Technical Details

### Heteroscedastic Regression
The model predicts both mean timing and uncertainty (log-variance) for each keystroke feature, enabling realistic variation in generated patterns.

### Conservative Numerical Bounds
- Log-variance clamped to `[-0.5, 0.5]` → variance ∈ `[0.6, 1.6]`
- Empirical variance bounds: `[0.5, 2.0]`  
- Variance ratio limits: `[0.1, 10.0]`

### Gradient Monitoring
Real-time detection of NaN/Inf gradients with immediate training termination to prevent model corruption.

## 📊 Dataset

**Source**: [KLiCKe Dataset](https://www.kaggle.com/datasets/julesking/tla-lab-pii-competition-dataset?resource=download-directory)

The dataset contains over 2,000 text-keystroke pairs with detailed timing information:
- **Dwell Time**: Key press duration
- **Flight Time**: Time between keystrokes  
- **Typing Speed**: Characters per minute
- **Keystroke Flags**: Letter, digit, punctuation, etc.

*Special thanks to the KLiCKe dataset contributors for making this research possible.*



## 🙏 Acknowledgments

- **Dataset**: [KLiCKe Competition Dataset](https://www.kaggle.com/datasets/julesking/tla-lab-pii-competition-dataset?resource=download-directory)
- **Model Architecture**: Microsoft DeBERTa-v3
- **Framework**: PyTorch & Hugging Face Transformers

---

<div align="center">

## Disclaimer:
    Credit to claude for making this readme look a lot nicer then I could 😊

</div>
