import os
import json
import torch
from torch.utils.data import Dataset
from dataPrepper import dataPrepper
import numpy as np

class dataLoader(Dataset):
   # loads up data from data/txts/*.txt and data/csv/*.csv
   # pairs together and preps for training
    def __init__(self, base_dir, prepper_class=dataPrepper, tokenizer=None,
                 max_length=512, preprocess=True, standardize=True, stats_file="cont_stats.json"):
        self.text_dir = os.path.join(base_dir, "texts")
        self.csv_dir  = os.path.join(base_dir, "csv")
        self.prepper_class = prepper_class
        self.tokenizer = tokenizer
        self.max_length = max_length   # only used to cap tokens at model max (e.g., 512)
        self.samples = []
        self.standardize = standardize
        self.cont_idx = [0, 1, 2]  # indices of continuous features: DwellTime, FlightTime, typing_speed
        self.stats_file = os.path.join(base_dir, stats_file)

        # check dirs exist
        if not os.path.isdir(self.text_dir):
            raise FileNotFoundError(f"Text dir not found: {self.text_dir}")
        if not os.path.isdir(self.csv_dir):
            raise FileNotFoundError(f"CSV dir not found: {self.csv_dir}")

        # --- load existing stats if available ---
        if self.standardize and os.path.isfile(self.stats_file):
            with open(self.stats_file, "r") as f:
                stats = json.load(f)
            self.cont_mean = torch.tensor(stats["mean"])
            self.cont_std  = torch.tensor(stats["std"])
            stats_loaded = True
            print(f"[INFO] Loaded existing continuous feature stats from {self.stats_file}")
        else:
            stats_loaded = False

        # --- load text/csv pairs ---
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

        # --- preprocess CSVs → raw feature arrays (variable length) ---
        kept = []
        all_cont_features = []
        if preprocess:
            for s in self.samples:
                try:
                    prep = prepper_class(s["csv_path"])
                    df = prep.get_prepared_data()  
                    keep_cols = [
                        "DwellTime", "FlightTime", "typing_speed",
                        "is_letter", "is_digit", "is_punct", "is_space",
                        "is_backspace", "is_enter", "is_shift"
                    ]
                    cols = [c for c in keep_cols if c in df.columns]
                    if not cols:
                        raise ValueError(f"No expected feature columns in {s['csv_path']}")
                    
                    feats = df[cols].to_numpy(dtype=float)  # [L_i, F], raw values
                    s["features"]  = feats
                    s["valid_len"] = feats.shape[0]

                    # collect continuous features for stats
                    if self.standardize and not stats_loaded:
                        all_cont_features.append(torch.tensor(feats[:, self.cont_idx], dtype=torch.float32))

                    kept.append(s)
                except Exception as e:
                    print(f"[WARN] Failed to preprocess {s['csv_path']}: {e}")
                    # Skip bad sample
            self.samples = kept
            if not self.samples:
                raise ValueError("No samples remained after preprocessing.")

            # compute and save stats to json
            if self.standardize and not stats_loaded and all_cont_features:
                all_cont = torch.cat(all_cont_features, dim=0)
                # Use nanmean and nanstd to handle NaN values (e.g., from FlightTime >10s cap)
                self.cont_mean = torch.tensor([all_cont[:, i][~torch.isnan(all_cont[:, i])].mean() for i in range(all_cont.shape[1])])
                self.cont_std  = torch.tensor([all_cont[:, i][~torch.isnan(all_cont[:, i])].std() for i in range(all_cont.shape[1])])
                # prevent division by zero
                self.cont_std[self.cont_std == 0] = 1.0
                # save to JSON
                stats = {
                    "mean": self.cont_mean.tolist(),
                    "std":  self.cont_std.tolist()
                }
                with open(self.stats_file, "w") as f:
                    json.dump(stats, f)
                print(f"[INFO] Saved continuous feature stats to {self.stats_file}")

        # when we don't standardize, set mean=0, std=1, dont rlly use this anymore but kept for legacy
        if not self.standardize:
            self.cont_mean = torch.zeros(len(self.cont_idx))
            self.cont_std  = torch.ones(len(self.cont_idx))

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
            enc = {k: v.squeeze(0) for k, v in enc.items()}  # remove batch dim

        features  = torch.tensor(s["features"], dtype=torch.float32)  # [L_i, F]
        valid_len = torch.tensor(s["valid_len"], dtype=torch.long)

        # standardize continuous features
        if self.standardize:
            features[:, self.cont_idx] = (features[:, self.cont_idx] - self.cont_mean) / self.cont_std

        return {
            "id": s["id"],
            "text": text,
            "input_ids": enc.get("input_ids"),          # 1D (T_i,) or None
            "attention_mask": enc.get("attention_mask"),
            "target": features,                         # [L_i, F] standardized features
            "target_len": valid_len
        }