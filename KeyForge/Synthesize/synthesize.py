"""Generate synthetic keystroke timing data from text using a trained model.

Tokenizes the input, runs the cached model bundle from
:func:`Synthesize.load_model.load_model`, samples per-character Dwell/Flight/
typing_speed from the predicted Gaussian distributions, clamps to physical
bounds, and returns a DataFrame (optionally writing a CSV).
"""
import contextlib
import os

import numpy as np
import pandas as pd
import torch

from config import (
    FEATURE_BOUNDS,
    FEATURE_COLUMNS,
    MAX_TOKEN_LENGTH,
    OUTPUT_DIR,
)
from Synthesize.load_model import load_model


# Re-export for backwards compatibility with existing imports.
DEFAULT_OUTPUT_DIR = OUTPUT_DIR


def predict_keystrokes(text, bundle=None, output_csv=None, **load_kwargs):
    """Predict keystroke timing features for every character in ``text``.

    Args:
        text (str): Raw text to synthesize keystrokes for.
        bundle (dict | None): Preloaded output of :func:`load_model`. If None,
            the model is loaded on every call (slow — prefer passing a cached
            bundle from the UI).
        output_csv (str | None): If provided, writes the DataFrame to this path.
        **load_kwargs: Forwarded to :func:`load_model` when ``bundle`` is None
            (``checkpoint_path``, ``base_model``, ``stats_path``, ``device``).

    Returns:
        pandas.DataFrame: Columns ``char, prev_char, DwellTime, FlightTime,
        typing_speed`` — one row per character (truncated to MAX_TOKEN_LENGTH).
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    text = text.strip()
    if not text:
        raise ValueError("text is empty")

    if bundle is None:
        bundle = load_model(**load_kwargs)

    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    cont_mean = bundle["cont_mean"]
    cont_std = bundle["cont_std"]
    device = bundle["device"]

    print(f"Synthesizing keystrokes for {len(text)} characters")

    #  Tokenize input text 
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_TOKEN_LENGTH,
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

    # Character identity tensor: explicit per-key signal for the model
    char_ids = torch.tensor(
        [ord(ch) % 256 for ch in text[:char_len]], dtype=torch.long, device=device
    ).unsqueeze(0)  # [1, T_char]

    enc = {k: v.to(device) for k, v in enc.items() if k in ["input_ids", "attention_mask"]}

    #  Run inference 
    # autocast only on CUDA; CPU autocast uses bfloat16 and clashes with the
    # float32 LayerNorm params shipped in the checkpoint.
    amp_ctx = torch.amp.autocast("cuda") if device.type == "cuda" else contextlib.nullcontext()
    with torch.no_grad(), amp_ctx:
        # Model outputs STANDARDIZED mean and log-variance
        mean_std, logvar_std = model(token_to_char_idx=token_to_char_idx, char_ids=char_ids, **enc)

        # Per-user affine calibration (trained in standardized space during
        # fine-tuning). Applied only when a user adapter is loaded.
        a_mean = bundle.get("a_mean")
        b_mean = bundle.get("b_mean")
        if a_mean is not None and b_mean is not None:
            mean_std = a_mean * mean_std + b_mean

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
    for idx, key in enumerate(["DwellTime", "FlightTime", "typing_speed"]):
        lo, hi = FEATURE_BOUNDS[key]
        continuous[:, :, idx] = torch.clamp(continuous[:, :, idx], min=lo, max=hi)

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
    df = pd.DataFrame(preds, columns=[
        "DwellTime", "FlightTime", "typing_speed",
    ])

    # Add character columns — ties each DwellTime to a char, each FlightTime to a char pair
    chars = list(text[:len(df)])
    prev_chars = [""] + chars[:-1]
    df.insert(0, "char", chars[:len(df)])
    df.insert(1, "prev_char", prev_chars[:len(df)])

    df = df[FEATURE_COLUMNS]

    if output_csv is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Saved predicted keystroke CSV: {output_csv}")
    print(df.head())
    return df
