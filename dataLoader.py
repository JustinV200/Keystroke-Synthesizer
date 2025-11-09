import os
import numpy as np
import torch
from torch.utils.data import Dataset
from dataPrepper import dataPrepper

class dataLoader(Dataset):
    def __init__(self, base_dir, prepper_class=dataPrepper, tokenizer=None, max_length=512, preprocess=True, max_keys=512):
        self.text_dir = os.path.join(base_dir, "texts")
        self.csv_dir = os.path.join(base_dir, "csv")
        self.prepper_class = prepper_class
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_keys = max_keys
        self.samples = []

        # Find matching files by ID
        text_files = [f for f in os.listdir(self.text_dir) if f.endswith(".txt")]
        csv_files = [f for f in os.listdir(self.csv_dir) if f.endswith(".csv")]

        text_ids = {os.path.splitext(f)[0] for f in text_files}
        csv_ids = {os.path.splitext(f)[0] for f in csv_files}
        matched_ids = sorted(list(text_ids & csv_ids))

        if not matched_ids:
            raise ValueError("No matching text/csv file pairs found.")

        for fid in matched_ids:
            self.samples.append({
                "id": fid,
                "txt_path": os.path.join(self.text_dir, f"{fid}.txt"),
                "csv_path": os.path.join(self.csv_dir, f"{fid}.csv")
            })

        # Preprocess each CSV into fixed-length feature tensors
        if preprocess:
            for s in self.samples:
                try:
                    prep = prepper_class(s["csv_path"], verbose=False)
                    features = prep.get_prepared_data(feature_only=True).values
                    num_features = features.shape[1]

                    # Pad/truncate to uniform length
                    if features.shape[0] > self.max_keys:
                        features = features[:self.max_keys, :]
                    elif features.shape[0] < self.max_keys:
                        pad = np.zeros((self.max_keys - features.shape[0], num_features))
                        features = np.vstack([features, pad])

                    s["features"] = features
                except Exception as e:
                    print(f"[WARN] Failed to preprocess {s['csv_path']}: {e}")
                    s["features"] = np.zeros((self.max_keys, 10))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load text
        with open(sample["txt_path"], "r", encoding="utf-8") as f:
            text = f.read().strip()

        # Tokenize for transformer
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
