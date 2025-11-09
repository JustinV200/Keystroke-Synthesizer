import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, get_scheduler
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from dataLoader import dataLoader  # your custom loader

# 1. configuration

BASE_MODEL = "deberta-v3-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 12
BATCH_SIZE = 8
LR = 2e-5
WEIGHT_DECAY = 0.01
MAX_TOKENS = 512
MAX_KEYS = 512
PATIENCE = 3    # early stopping patience (epochs)
OUTPUT_DIR = "checkpoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# TensorBoard
writer = SummaryWriter("runs/keystroke_experiment")

# 2. Datasets and DataLoaders

print("Loading tokenizer and datasets...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

train_dataset = dataLoader(
    base_dir="data/train", tokenizer=tokenizer,
    preprocess=True, max_length=MAX_TOKENS, max_keys=MAX_KEYS
)
val_dataset = dataLoader(
    base_dir="data/val", tokenizer=tokenizer,
    preprocess=True, max_length=MAX_TOKENS, max_keys=MAX_KEYS
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

# 3. Models

class TextToKeystrokeModel(nn.Module):
    def __init__(self, base_model, num_features):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size

        # deeper regression head with normalization and dropout
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
        hidden_states = x.last_hidden_state  # (B, L, H)
        preds = self.regressor(hidden_states)  # (B, L, num_features)
        return preds


# get feature dimension from dataset
num_features = train_dataset[0]["target"].shape[1]
model = TextToKeystrokeModel(BASE_MODEL, num_features).to(DEVICE)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)  # multi-GPU support

#Optimizer, Loss, Scheduler

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loss_fn = nn.MSELoss()


scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2)

scaler = GradScaler()

#traning loop
best_val_loss = float("inf")
patience_counter = 0
global_step = 0

print("Starting training...")
for epoch in range(EPOCHS):
    # ---------------- TRAIN ----------------
    model.train()
    total_train_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        target = batch["target"].to(DEVICE)

        optimizer.zero_grad()
        with autocast():  # mixed precision
            preds = model(input_ids, mask)
            loss = loss_fn(preds, target)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step(epoch + len(progress_bar) / len(train_loader))

        total_train_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
        global_step += 1
        writer.add_scalar("Loss/train", loss.item(), global_step)

    avg_train_loss = total_train_loss / len(train_loader)

    # ---------------- VALIDATION ----------------
    model.eval()
    val_loss, val_mae = 0.0, 0.0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            target = batch["target"].to(DEVICE)

            preds = model(input_ids, mask)
            loss = loss_fn(preds, target)
            mae = (preds - target).abs().mean().item()

            val_loss += loss.item()
            val_mae += mae

    val_loss /= len(val_loader)
    val_mae /= len(val_loader)

    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("MAE/val", val_mae, epoch)

    print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.5f}, Val Loss={val_loss:.5f}, Val MAE={val_mae:.5f}")

    # ---------------- CHECKPOINT ----------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        save_path = os.path.join(OUTPUT_DIR, "best_model.pt")
        torch.save(model.state_dict(), save_path)
        print(f"Saved new best model to {save_path}")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

writer.close()
print("Training complete!")
