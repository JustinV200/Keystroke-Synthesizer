import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from torch import nn
from torch.cuda.amp import autocast
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
    num_features=15,  # same as your training feature count
    max_length=512,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load text
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    print(f"Loaded text ({len(text)} characters)")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = TextToKeystrokeModel(base_model, num_features).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Tokenize
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    # Predict
    with torch.no_grad(), autocast():
        preds = model(**enc).cpu().numpy()[0]  # (seq_len, num_features)

    # Create DataFrame
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
