# synthesizeKeystrokes.py
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from torch import nn
from collections import OrderedDict


class TextToKeystrokeModelMultiHead(nn.Module):
    def __init__(self, base_model, num_continuous=6, num_flags=9):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size

        self.backbone = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.regression_head = nn.Linear(256, num_continuous)
        self.classification_head = nn.Linear(256, num_flags)

    def forward(self, input_ids, attention_mask):
        x = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        shared = self.backbone(x.last_hidden_state)
        return self.regression_head(shared), self.classification_head(shared)


def predict_keystrokes(
    text_path,
    checkpoint_path="checkpoints/best_model.pt",
    base_model="microsoft/deberta-v3-base",
    output_csv="predicted_keystrokes.csv",
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
    num_continuous, num_flags = 6, 9
    model = TextToKeystrokeModelMultiHead(base_model, num_continuous, num_flags).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    # handle DataParallel checkpoints
    if any(k.startswith("module.") for k in checkpoint.keys()):
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        checkpoint = new_state_dict

    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    # Tokenize input
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    enc = {k: v.to(device) for k, v in enc.items() if k in ["input_ids", "attention_mask"]}

    # Run inference
    with torch.no_grad(), torch.amp.autocast("cuda" if device.type == "cuda" else "cpu"):
        continuous, logits = model(**enc)
        flags = torch.sigmoid(logits)  # probabilities 0–1

        # Assemble full feature tensor
        B, T, _ = continuous.shape
        out = torch.zeros(B, T, 15, device=continuous.device)
        cont_idx = [0, 1, 2, 3, 13, 14]
        flag_idx = [4, 5, 6, 7, 8, 9, 10, 11, 12]

        out[:, :, cont_idx] = continuous
        out[:, :, flag_idx] = flags
        preds = out.cpu().numpy()[0]

    # Trim predictions to actual text length
    seq_len = enc["attention_mask"].sum().item()
    preds = preds[:seq_len]

    # Save to CSV
    feature_cols = [
        "DwellTime", "FlightTime", "typing_speed", "char_code",
        "is_letter", "is_digit", "is_punct", "is_space",
        "is_backspace", "is_enter", "is_shift",
        "is_pause_2s", "is_pause_5s", "cum_backspace", "cum_chars"
    ]
    df = pd.DataFrame(preds, columns=feature_cols)
    df.to_csv(output_csv, index=False)

    print(f"Saved predicted keystroke CSV: {output_csv}")
    print(df.head())


if __name__ == "__main__":
    predict_keystrokes(
        text_path="sample.txt",
        checkpoint_path="checkpoints/best_model.pt",
        base_model="microsoft/deberta-v3-base",
        output_csv="predicted_keystrokes.csv"
    )
