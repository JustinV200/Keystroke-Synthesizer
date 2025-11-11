import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
from torch import amp
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from dataLoader import dataLoader

# ---------------- Config ----------------
BASE_MODEL   = "microsoft/deberta-v3-base"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS       = 12
BATCH_SIZE   = 8
LR           = 2e-5
WEIGHT_DECAY = 0.01
MAX_TOKENS   = 512
PATIENCE     = 3
OUTPUT_DIR   = "checkpoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)
writer = SummaryWriter("runs/keystroke_experiment")
torch.set_float32_matmul_precision("high")

# ---------------- Batch collate (no target padding) ----------------
def make_collate_fn(pad_token_id: int):
    def collate(batch):
        # Token side (pad for transformer)
        input_ids = [b["input_ids"] for b in batch]
        attn_mask = [b["attention_mask"] for b in batch]
        max_t = max(t.shape[0] for t in input_ids)
        input_ids = [F.pad(t, (0, max_t - t.shape[0]), value=pad_token_id) for t in input_ids]
        attn_mask = [F.pad(m, (0, max_t - m.shape[0]), value=0) for m in attn_mask]
        input_ids = torch.stack(input_ids, dim=0)
        attn_mask = torch.stack(attn_mask, dim=0)

        # Target side (keep variable-length sequences)
        targets = [b["target"] for b in batch]
        target_masks = [torch.ones_like(t[:, :1]) for t in targets]  # simple validity mask

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "target": targets,          # list of [L_i, F]
            "target_mask": target_masks # list of [L_i, 1]
        }
    return collate

# ---------------- Data ----------------
print("Loading tokenizer and datasets...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

full_dataset = dataLoader(
    base_dir="data", tokenizer=tokenizer,
    preprocess=True, max_length=MAX_TOKENS
)

n_total = len(full_dataset)
n_val   = int(0.2 * n_total)
n_train = n_total - n_val
train_dataset, val_dataset = random_split(
    full_dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)

pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
collate_fn = make_collate_fn(pad_id)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=True, collate_fn=collate_fn)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

# ---------------- Model ----------------
class TextToKeystrokeModel(nn.Module):
    def __init__(self, base_model, num_features):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(hidden, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256),   nn.ReLU(),
            nn.Linear(256, num_features)
        )

    def forward(self, input_ids, attention_mask):
        x = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.regressor(x.last_hidden_state)   # [B, T, F]

# infer feature size from one batch
probe = next(iter(train_loader))
num_features = probe["target"][0].shape[-1]

model = TextToKeystrokeModel(BASE_MODEL, num_features).to(DEVICE)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2)
scaler    = amp.GradScaler(device="cuda" if DEVICE.type == "cuda" else "cpu")

# ---------------- Train ----------------
best_val, patience, step = float("inf"), 0, 0
print("Starting training...")

for epoch in range(EPOCHS):
    model.train()
    train_loss_sum = 0.0
    for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
        input_ids    = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_m  = batch["attention_mask"].to(DEVICE, non_blocking=True)
        targets      = [t.to(DEVICE, non_blocking=True) for t in batch["target"]]

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
            preds = model(input_ids, attention_m)  # [B, Lt, F]
            loss_sum, count = 0.0, 0
            for j, target in enumerate(targets):
                L = min(preds.shape[1], target.shape[0])
                se = (preds[j, :L, :] - target[:L, :]).pow(2)
                loss_sum += se.sum()
                count += L
            loss = loss_sum / max(1, count)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step(epoch + (i + 1) / max(1, len(train_loader)))

        train_loss_sum += loss.item()
        step += 1
        writer.add_scalar("Loss/train", loss.item(), step)

    avg_train = train_loss_sum / max(1, len(train_loader))

    # ---- Validation ----
    model.eval()
    val_loss_sum, val_mae_sum, val_count = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids    = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_m  = batch["attention_mask"].to(DEVICE, non_blocking=True)
            targets      = [t.to(DEVICE, non_blocking=True) for t in batch["target"]]

            with amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                preds = model(input_ids, attention_m)
                for j, target in enumerate(targets):
                    L = min(preds.shape[1], target.shape[0])
                    se = (preds[j, :L, :] - target[:L, :]).pow(2)
                    ae = (preds[j, :L, :] - target[:L, :]).abs()
                    val_loss_sum += se.sum().item()
                    val_mae_sum  += ae.sum().item()
                    val_count    += L

    val_loss = val_loss_sum / max(1, val_count)
    val_mae  = val_mae_sum  / max(1, val_count)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("MAE/val",  val_mae,  epoch)
    print(f"Epoch {epoch+1}: Train={avg_train:.5f}  Val={val_loss:.5f}  MAE={val_mae:.5f}")

    if val_loss < best_val:
        best_val = val_loss
        patience = 0
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))
        print("Saved new best model.")
    else:
        patience += 1
        if patience >= PATIENCE:
            print("Early stopping triggered.")
            break

writer.close()
print("Training complete!")
