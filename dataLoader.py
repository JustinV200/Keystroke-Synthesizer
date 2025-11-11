import os
import numpy as np
import torch
from torch.utils.data import Dataset
from dataPrepper import dataPrepper

class dataLoader(Dataset):
   #loads up data from data/txts/*.txt and data/csv/*.csv
   #pairs together and preps for training
    def __init__(self, base_dir, prepper_class=dataPrepper, tokenizer=None,
                 max_length=512, preprocess=True):
        self.text_dir = os.path.join(base_dir, "texts")
        self.csv_dir  = os.path.join(base_dir, "csv")
        self.prepper_class = prepper_class
        self.tokenizer = tokenizer
        self.max_length = max_length   # only used to cap tokens at model max (e.g., 512)
        self.samples = []

        if not os.path.isdir(self.text_dir):
            raise FileNotFoundError(f"Text dir not found: {self.text_dir}")
        if not os.path.isdir(self.csv_dir):
            raise FileNotFoundError(f"CSV dir not found: {self.csv_dir}")

        text_ids = {os.path.splitext(f)[0] for f in os.listdir(self.text_dir) if f.endswith(".txt")}
        csv_ids  = {os.path.splitext(f)[0] for f in os.listdir(self.csv_dir)  if f.endswith(".csv")}
        matched_ids = sorted(text_ids & csv_ids)
        if not matched_ids:
            raise ValueError("No matching text/csv file pairs found.")

        for fid in matched_ids:
            self.samples.append({
                "id": fid,
                "txt_path": os.path.join(self.text_dir, f"{fid}.txt"),
                "csv_path": os.path.join(self.csv_dir,  f"{fid}.csv")
            })

        # Preprocess CSVs → raw feature arrays (variable length)
        if preprocess:
            kept = []
            for s in self.samples:
                try:
                    prep = prepper_class(s["csv_path"])
                    df = prep.get_prepared_data()  # ensure dataPrepper.normalize() is disabled
                    keep_cols = [
                        "DwellTime", "FlightTime", "typing_speed", "char_code",
                        "is_letter", "is_digit", "is_punct", "is_space",
                        "is_backspace", "is_enter", "is_shift",
                        "is_pause_2s", "is_pause_5s", "cum_backspace", "cum_chars"
                    ]
                    cols = [c for c in keep_cols if c in df.columns]
                    if not cols:
                        raise ValueError(f"No expected feature columns in {s['csv_path']}")

                    feats = df[cols].to_numpy()  # [L_i, F], raw values
                    s["features"]  = feats
                    s["valid_len"] = feats.shape[0]
                    kept.append(s)
                except Exception as e:
                    print(f"[WARN] Failed to preprocess {s['csv_path']}: {e}")
                    # Skip bad sample
            self.samples = kept
            if not self.samples:
                raise ValueError("No samples remained after preprocessing.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Load full text
        with open(s["txt_path"], "r", encoding="utf-8") as f:
            text = f.read().strip()

        # Tokenize (variable length; do NOT pad here)
        enc = {}
        if self.tokenizer is not None:
            enc = self.tokenizer(
                text,
                return_tensors="pt",
                padding=False,          # natural token length
                truncation=True,        # cap to model max length (e.g., 512)
                max_length=self.max_length,
            )
            enc = {k: v.squeeze(0) for k, v in enc.items()}  # -> 1D tensors

        features  = torch.tensor(s["features"],  dtype=torch.float32)  # [L_i, F]
        valid_len = torch.tensor(s["valid_len"], dtype=torch.long)

        return {
            "id": s["id"],
            "text": text,
            "input_ids": enc.get("input_ids"),          # 1D (T_i,) or None
            "attention_mask": enc.get("attention_mask"),
            "target": features,                         # [L_i, F] raw features
            "target_len": valid_len
        }
