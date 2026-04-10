# synthesizeKeystrokes.py
import torch
import pandas as pd
import json
import sys
import os
from transformers import AutoTokenizer, AutoModel
from torch import nn
from collections import OrderedDict
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Trainer.TextToKeystrokeModelMultiHead import TextToKeystrokeModelMultiHead


def predict_keystrokes(
    text_path,
    checkpoint_path="checkpoints/best_model.pt",
    base_model="microsoft/deberta-v3-base",
    output_csv="predicted_keystrokes.csv",
    stats_path="data/cont_stats.json",
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #  Load text 
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    print(f"Loaded text ({len(text)} characters)")

    #  Load tokenizer and model 
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    num_continuous = 3
    model = TextToKeystrokeModelMultiHead(base_model, num_continuous).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # handle DataParallel checkpoints
    if any(k.startswith("module.") for k in checkpoint.keys()):
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        checkpoint = new_state_dict

    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    #  Load standardization stats 
    with open(stats_path, "r") as f:
        stats = json.load(f)

    cont_mean = torch.tensor(stats["mean"], device=device)
    cont_std  = torch.tensor(stats["std"],  device=device)

    #  Tokenize input text 
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
        return_offsets_mapping=True,
    )

    # Build character-to-token mapping for character-level predictions
    offset_mapping = enc["offset_mapping"].squeeze(0)  # [T, 2]
    max_char_covered = 0
    for i in range(offset_mapping.shape[0]):
        end_val = offset_mapping[i, 1].item()
        if end_val > max_char_covered:
            max_char_covered = end_val
    char_len = min(len(text), max_char_covered)

    token_to_char_idx = torch.zeros(char_len, dtype=torch.long, device=device)
    for tok_idx in range(offset_mapping.shape[0]):
        start = offset_mapping[tok_idx, 0].item()
        end = offset_mapping[tok_idx, 1].item()
        if start == end:  # special token (CLS, SEP, PAD)
            continue
        for c in range(start, min(end, char_len)):
            token_to_char_idx[c] = tok_idx
    token_to_char_idx = token_to_char_idx.unsqueeze(0)  # [1, T_char]

    enc = {k: v.to(device) for k, v in enc.items() if k in ["input_ids", "attention_mask"]}

    #  Run inference 
    with torch.no_grad(), torch.amp.autocast("cuda" if device.type == "cuda" else "cpu"):
        # Model outputs STANDARDIZED mean and log-variance
        mean_std, logvar_std = model(token_to_char_idx=token_to_char_idx, **enc)

        #  De-standardize mean and variance 
        # De-standardize mean: y_mean = y_std * std + mean
        mean = mean_std * cont_std + cont_mean
        
        # De-standardize variance: var = exp(logvar_std) * std^2
        variance_std = torch.exp(logvar_std)
        variance = variance_std * (cont_std ** 2)
        std = torch.sqrt(variance.clamp(min=1e-8))  # prevent negative/zero variance
    
        #  Sample from predicted distributions 
        # N(mean, std) - adds realistic variability!
        continuous = torch.randn_like(mean) * std + mean

    # force outputs to float32
    continuous = continuous.float()

    #  Physical constraints (match training data preprocessing)
    continuous[:, :, 0] = torch.clamp(continuous[:, :, 0], min=0.0, max=300.0)  # DwellTime (matches dataPrepper cap)
    continuous[:, :, 1] = torch.clamp(continuous[:, :, 1], min=0.0, max=900.0)  # FlightTime (matches dataPrepper cap)
    continuous[:, :, 2] = torch.clamp(continuous[:, :, 2], min=0.0, max=490.0)  # typing_speed (matches dataPrepper cap)

#  Assemble output 
    B, T, _ = continuous.shape
    preds = continuous.cpu().numpy()[0]

    # Trim predictions to actual character length
    preds = preds[:char_len]
    
    # fix first flight time
    # FlightTime for first keystroke is undefined (no previous key)
    if len(preds) > 0:
        preds[0, 1] = np.nan  # Set first FlightTime to NaN

    # Save to CSV
    feature_cols = [
        "char", "prev_char",
        "DwellTime", "FlightTime", "typing_speed",
    ]

    df = pd.DataFrame(preds, columns=[
        "DwellTime", "FlightTime", "typing_speed",
    ])

    # Add character columns — ties each DwellTime to a char, each FlightTime to a char pair
    chars = list(text[:len(df)])
    prev_chars = [""] + chars[:-1]
    df.insert(0, "char", chars[:len(df)])
    df.insert(1, "prev_char", prev_chars[:len(df)])

    df = df[feature_cols] 

    df.to_csv(output_csv, index=False)

    print(f"Saved predicted keystroke CSV: {output_csv}")
    print(df.head())

if __name__ == "__main__":
    predict_keystrokes(
        text_path="sample.txt",
        checkpoint_path="checkpoints/best_model.pt",
        base_model="microsoft/deberta-v3-base",
        output_csv="predicted_keystrokes.csv",
        stats_path="data/cont_stats.json"
    )
