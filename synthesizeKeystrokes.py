import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from torch import nn
from collections import OrderedDict
import os


class TextToKeystrokeModel(nn.Module):
    def __init__(self, base_model, num_features):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_features)
        )

    def forward(self, input_ids, attention_mask):
        x = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = x.last_hidden_state
        preds = self.regressor(hidden_states)
        return preds


def predict_keystrokes(
    text_path,
    checkpoint_path="checkpoints/best_model.pt",
    base_model="microsoft/deberta-v3-base",
    output_csv="predicted_keystrokes.csv",
    num_features=15,
    max_length=512,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    print(f"Loaded text ({len(text)} characters)")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = TextToKeystrokeModel(base_model, num_features).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if any(k.startswith("module.") for k in checkpoint.keys()):
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        checkpoint = new_state_dict
    model.load_state_dict(checkpoint, strict=False)

    model.eval()

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    enc = {k: v.to(device) for k, v in enc.items() if k in ["input_ids", "attention_mask"]}

    with torch.no_grad(), torch.amp.autocast("cuda"):
        preds = model(**enc).cpu().numpy()[0]

    feature_cols = [
        "DwellTime", "FlightTime", "typing_speed", "char_code",
        "is_letter", "is_digit", "is_punct", "is_space",
        "is_backspace", "is_enter", "is_shift",
        "is_pause_2s", "is_pause_5s", "cum_backspace", "cum_chars"
    ][:num_features]

    df = pd.DataFrame(preds, columns=feature_cols)
    df.to_csv(output_csv, index=False)

    print(f"Saved predicted keystroke CSV: {output_csv}")
    print(df.head())


if __name__ == "__main__":
    predict_keystrokes(
        text_path="sample.txt",
        checkpoint_path="checkpoints/best_model.pt",
        base_model="microsoft/deberta-v3-base",
        output_csv="predicted_keystrokes.csv",
        num_features=15
    )
