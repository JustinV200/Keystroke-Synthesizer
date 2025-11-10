import os
import numpy as np
import torch
from torch.utils.data import Dataset
from dataPrepper import dataPrepper

class dataLoader(Dataset):
    def __init__(self, base_dir, prepper_class=dataPrepper, tokenizer=None,
                 max_length=512, preprocess=True, max_keys=512):
        self.text_dir = os.path.join(base_dir, "texts")
        self.csv_dir  = os.path.join(base_dir, "csv")
        self.prepper_class = prepper_class
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_keys = max_keys
        self.samples = []

        # Sanity: dirs must exist
        if not os.path.isdir(self.text_dir):
            raise FileNotFoundError(f"Text dir not found: {self.text_dir}")
        if not os.path.isdir(self.csv_dir):
            raise FileNotFoundError(f"CSV dir not found: {self.csv_dir}")

        # Pair files by shared stem
        text_files = [f for f in os.listdir(self.text_dir) if f.endswith(".txt")]
        csv_files  = [f for f in os.listdir(self.csv_dir)  if f.endswith(".csv")]
        text_ids = {os.path.splitext(f)[0] for f in text_files}
        csv_ids  = {os.path.splitext(f)[0] for f in csv_files}
        matched_ids = sorted(text_ids & csv_ids)
        if not matched_ids:
            raise ValueError("No matching text/csv file pairs found.")

        for fid in matched_ids:
            self.samples.append({
                "id": fid,
                "txt_path": os.path.join(self.text_dir, f"{fid}.txt"),
                "csv_path": os.path.join(self.csv_dir,  f"{fid}.csv")
            })

        # Preprocess CSVs to fixed-length feature arrays
        if preprocess:
            kept = []
            for s in self.samples:
                try:
                    prep = prepper_class(s["csv_path"])
                    df = prep.get_prepared_data()  # no feature_only kwarg

                    # Select model features (only those present)
                    keep_cols = [
                        "DwellTime", "FlightTime", "typing_speed", "char_code",
                        "is_letter", "is_digit", "is_punct", "is_space",
                        "is_backspace", "is_enter", "is_shift",
                        "is_pause_2s", "is_pause_5s", "cum_backspace", "cum_chars"
                    ]
                    cols = [c for c in keep_cols if c in df.columns]
                    if not cols:
                        raise ValueError(f"No expected feature columns in {s['csv_path']}")

                    features = df[cols].to_numpy()
                    num_features = features.shape[1]

                    # Pad/truncate to max_keys
                    if features.shape[0] > self.max_keys:
                        features = features[:self.max_keys, :]
                    elif features.shape[0] < self.max_keys:
                        pad = np.zeros((self.max_keys - features.shape[0], num_features),
                                       dtype=features.dtype)
                        features = np.vstack([features, pad])

                    s["features"] = features
                    kept.append(s)
                except Exception as e:
                    print(f"[WARN] Failed to preprocess {s['csv_path']}: {e}")
                    # skip bad sample

            self.samples = kept
            if not self.samples:
                raise ValueError("No samples remained after preprocessing.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load text
        with open(sample["txt_path"], "r", encoding="utf-8") as f:
            text = f.read().strip()

        # Tokenize (if provided)
        if self.tokenizer:
            enc = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            enc = {k: v.squeeze(0) for k, v in enc.items()}
        else:
            enc = {}

        features = torch.tensor(sample["features"], dtype=torch.float32)

        return {
            "id": sample["id"],
            "input_ids": enc.get("input_ids"),
            "attention_mask": enc.get("attention_mask"),
            "text": text,
            "target": features
        }
