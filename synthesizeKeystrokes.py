# synthesizeKeystrokes.py
import torch
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModel
from torch import nn
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np

class TextToKeystrokeModelMultiHead(nn.Module):
    def __init__(self, base_model, num_continuous=3, num_flags=9):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # Heteroscedastic regression heads - predict mean AND variance
        self.mean_head = nn.Linear(256, num_continuous)
        self.logvar_head = nn.Linear(256, num_continuous)

        # Classification head (binary flags)
        self.classification_head = nn.Linear(256, num_flags)

    def forward(self, input_ids, attention_mask):
        x = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        shared = self.backbone(x.last_hidden_state)

        mean = self.mean_head(shared)
        logvar = self.logvar_head(shared)
        logits = self.classification_head(shared)
        
        return mean, logvar, logits


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
    num_continuous, num_flags = 3, 9
    model = TextToKeystrokeModelMultiHead(base_model, num_continuous, num_flags).to(device)

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
    )
    enc = {k: v.to(device) for k, v in enc.items() if k in ["input_ids", "attention_mask"]}

    #  Run inference 
    with torch.no_grad(), torch.amp.autocast("cuda" if device.type == "cuda" else "cpu"):
        # Model outputs STANDARDIZED mean and log-variance
        mean_std, logvar_std, logits = model(**enc)

        # Convert logits → probabilities for binary flags
        flags = torch.sigmoid(logits)  # values in [0, 1]

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
    flags = flags.float()

    # Undo log transform for FlightTime (index 1) - converts from log-space back to milliseconds
    continuous[:, :, 1] = torch.exp(continuous[:, :, 1]) - 1

    #  Physical constraints 
    #todo, find actual minimums
    continuous[:, :, 0] = torch.clamp(continuous[:, :, 0], min=20.0, max=300.0)  # DwellTime (cap at 300ms to match training)
    continuous[:, :, 1] = torch.clamp(continuous[:, :, 1], min=10.0, max=900.0)  # FlightTime (cap at 900ms to match training)
    continuous[:, :, 2] = torch.clamp(continuous[:, :, 2], min=100.0, max=500.0)  # typing_speed (realistic minimum)

#  Assemble full feature tensor 
    B, T, _ = continuous.shape
    out = torch.zeros(B, T, 12, device=continuous.device)

    # Indices must match training
    cont_idx = [0, 1, 2]  # 3 continuous outputs: DwellTime, FlightTime, typing_speed
    flag_idx = [3, 4, 5, 6, 7, 8, 9, 10, 11]  # 9 binary flags

    out[:, :, cont_idx] = continuous
    out[:, :, flag_idx] = flags

    preds = out.cpu().numpy()[0]

    # Trim predictions to actual text length
    seq_len = enc["attention_mask"].sum().item()
    preds = preds[:seq_len]
    
    # fix first flight time
    # FlightTime for first keystroke is undefined (no previous key)
    if len(preds) > 0:
        preds[0, 1] = np.nan  # Set first FlightTime to NaN

    # Save to CSV
    feature_cols = [
        "DwellTime", "FlightTime", "typing_speed",
        "is_letter", "is_digit", "is_punct", "is_space",
        "is_backspace", "is_enter", "is_shift",
        "is_pause_2s", "is_pause_5s"
    ]

    df = pd.DataFrame(preds, columns=[
        "DwellTime", "FlightTime", "typing_speed",
        "is_letter", "is_digit", "is_punct", "is_space",
        "is_backspace", "is_enter", "is_shift",
        "is_pause_2s", "is_pause_5s"
    ])
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
