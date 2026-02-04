# Training Configuration
# Justin Verlin, Keystroke Synthesizer
# config.py - Centralized configuration for training

import torch
import os

# Model Configuration
BASE_MODEL   = "microsoft/deberta-v3-base"
MAX_TOKENS   = 512 # max tokens for transformer input

# Training Configuration  
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu") #use gpu if possible
EPOCHS       = 12 # number of runs
BATCH_SIZE   = 8 # per-GPU batch size
LR           = 1e-5 # learning rate (reduced for stability)
WEIGHT_DECAY = 0.01 # weight decay, for how much to regularize
PATIENCE     = 3 # early stopping patience, if no val improvement

# KL Regularization Configuration
KL_WEIGHT_START = 0.001   # KL weight at epoch 0 (focus on mean first)
KL_WEIGHT_END   = 0.03   # KL weight at final annealing epoch (then focus on variance)
KL_ANNEAL_EPOCHS = 8    # Linearly increase KL weight over first 6 epochs

# Feature-specific KL multipliers [DwellTime, FlightTime, typing_speed]
KL_FEATURE_WEIGHTS = [1.0, 0, .3]  # feature specific weights for KL divergence
#KL_FEATURE_WEIGHTS = [2.0, 3.0, 0.5] #old kl weights

# Output Configuration
OUTPUT_DIR   = "checkpoints" # where to save models
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Performance Optimization
torch.set_float32_matmul_precision("high")
