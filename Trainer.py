#Justin Verlin, Keystroke Synthesizer Model Trainer
#Trainer.py
# 1/4/2026
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
from torch import amp
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from dataLoader import dataLoader
import sys
# config and init
BASE_MODEL   = "microsoft/deberta-v3-base"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu") #use gpu if possible
EPOCHS       = 12 # number of runs
BATCH_SIZE   = 8 # per-GPU batch size
LR           = 2e-5 # learning rate
WEIGHT_DECAY = 0.01 # weight decay, for how much to regularize
KL_WEIGHT_START = 0.001   # KL weight at epoch 0 (focus on mean first)
KL_WEIGHT_END   = 0.03   # KL weight at final annealing epoch (then focus on variance)
KL_ANNEAL_EPOCHS = 8    # Linearly increase KL weight over first 6 epochs
# Feature-specific KL multipliers [DwellTime, FlightTime, typing_speed]
KL_FEATURE_WEIGHTS = [1.0, 0, .3]  # feature specific weights for KL divergence
#KL_FEATURE_WEIGHTS = [2.0, 3.0, 0.5] #old kl wieghts
MAX_TOKENS   = 512 # max tokens for transformer input
PATIENCE     = 3 # early stopping patience, if no val improvement
OUTPUT_DIR   = "checkpoints" # where to save models
os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.set_float32_matmul_precision("high")

# this will pad the variable-length inputs and targets in each batch
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
        target_masks = [torch.ones_like(t[:, :1]) for t in targets]  #simple validity mask

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "target": targets,          # list of [L_i, F]
            "target_mask": target_masks # list of [L_i, 1]
        }
    return collate

# Data loading
print("Loading tokenizer and datasets...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

full_dataset = dataLoader(
    base_dir="data", tokenizer=tokenizer,
    preprocess=True, max_length=MAX_TOKENS
)
# split into train/val
n_total = len(full_dataset)
n_val   = int(0.2 * n_total) # 20% for validation
n_train = n_total - n_val # rest for training

# random split with fixed seed for reproducibility
train_dataset, val_dataset = random_split(
    full_dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)
pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
collate_fn = make_collate_fn(pad_id)
# Data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=True, collate_fn=collate_fn)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

# Compute empirical variance from training data, used for kl weights, train towards realistic uncertainty estimates
print("Computing empirical variance from training data...")
all_cont_features = []
for idx in range(min(len(train_dataset), 2000)):  # Sample 2000
    sample = train_dataset[idx]
    all_cont_features.append(sample["target"][:, [0, 1, 2]])  # DwellTime, FlightTime, typing_speed
all_cont = torch.cat(all_cont_features, dim=0)
# Use NaN-aware variance computation (same approach as dataLoader)
empirical_var = torch.tensor([all_cont[:, i][~torch.isnan(all_cont[:, i])].var(unbiased=True) for i in range(all_cont.shape[1])])
# Ensure empirical variance is never too small to prevent numerical issues
empirical_var = torch.clamp(empirical_var, min=1e-4)
print(f"Empirical variance (standardized space): {empirical_var.tolist()}")
empirical_var = empirical_var.to(DEVICE)

# Create feature-specific KL weight tensor
kl_feature_weights = torch.tensor(KL_FEATURE_WEIGHTS, device=DEVICE)  # [3]
print(f"Feature-specific KL weights: {KL_FEATURE_WEIGHTS}")

# Model definition, updated to have multi-head outputs, once for
# continuous features (regression), once for binary flags (classification)
class TextToKeystrokeModelMultiHead(nn.Module):
    def __init__(self, base_model, num_continuous=3, num_flags=9):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(hidden, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU()
        )
        
        # Heteroscedastic regression heads - predict mean AND variance
        self.mean_head = nn.Linear(256, num_continuous)
        self.logvar_head = nn.Linear(256, num_continuous)  # log-variance for numerical stability
        
        # Initialize logvar head to predict small variance initially
        nn.init.constant_(self.logvar_head.weight, 0.0)
        nn.init.constant_(self.logvar_head.bias, -2.0)  # exp(-2) ≈ 0.135 initial variance
        
        # Classification head (binary flags)
        self.classification_head = nn.Linear(256, num_flags)

    def forward(self, input_ids, attention_mask):
        x = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = x.last_hidden_state  # [B, T, hidden]
        
        shared = self.backbone(hidden)  # [B, T, 256]
        
        mean = self.mean_head(shared)  # [B, T, num_continuous]
        logvar = self.logvar_head(shared)  # [B, T, num_continuous]
        logits = self.classification_head(shared)  # [B, T, num_flags]
        return mean, logvar, logits

# infer feature size from one batch
probe = next(iter(train_loader))
num_features = probe["target"][0].shape[-1]

# Initialize multi-head model with correct feature counts:
# - 3 continuous features: DwellTime, FlightTime, typing_speed
# - 7 binary flags: is_letter, is_digit, is_punct, is_space, is_backspace, is_enter, is_shift
model = TextToKeystrokeModelMultiHead(BASE_MODEL, num_continuous=3, num_flags=7).to(DEVICE)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2)
scaler    = amp.GradScaler(device="cuda" if DEVICE.type == "cuda" else "cpu")

#  Training loop
best_val, patience, step = float("inf"), 0, 0
print("Starting training...")
#define which indices are continuous and which are binary flags
cont_idx = [0, 1, 2]  # DwellTime, FlightTime, typing_speed
flag_idx = [3, 4, 5, 6, 7, 8, 9]  # 7 binary flags


for epoch in range(EPOCHS):
    # Compute current KL weight using linear annealing schedule
    if epoch < KL_ANNEAL_EPOCHS:
        kl_weight = KL_WEIGHT_START + (KL_WEIGHT_END - KL_WEIGHT_START) * (epoch / KL_ANNEAL_EPOCHS)
    else:
        kl_weight = KL_WEIGHT_END
    
    print(f"Epoch {epoch+1}/{EPOCHS} - KL weight: {kl_weight:.5f}")
    
    model.train()
    train_loss_sum = 0.0
    for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
        input_ids    = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_m  = batch["attention_mask"].to(DEVICE, non_blocking=True)
        targets      = [t.to(DEVICE, non_blocking=True) for t in batch["target"]]

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
            # Debug: Check for NaN in inputs
            if torch.isnan(input_ids).any():
                print(f"  WARNING: NaN in input_ids for batch {i}")
                continue
            if torch.isnan(attention_m).any():
                print(f"  WARNING: NaN in attention_mask for batch {i}")
                continue
            for j, target in enumerate(targets):
                if torch.isnan(target).any():
                    print(f"  WARNING: NaN in target {j} for batch {i}")
                    print(f"    Target shape: {target.shape}")
                    print(f"    NaN locations: {torch.isnan(target).sum(dim=0)}")
            
            # Model now returns mean, logvar, logits
            mean, logvar, logits = model(input_ids, attention_m)
            
            # Debug: Check for NaN in model outputs
            if torch.isnan(mean).any():
                print(f"  WARNING: NaN in mean output for batch {i}")
            if torch.isnan(logvar).any():
                print(f"  WARNING: NaN in logvar output for batch {i}")
                print(f"    logvar range: min={logvar.min():.3f}, max={logvar.max():.3f}")
            if torch.isnan(logits).any():
                print(f"  WARNING: NaN in logits output for batch {i}")
            
            loss_sum, bce_sum, count = 0.0, 0.0, 0
            for j, target in enumerate(targets):
                L = min(mean.shape[1], target.shape[0])

                # Create mask for valid (non-NaN) continuous features
                valid_mask = ~torch.isnan(target[:L, cont_idx])  # [L, 3]

                # Gaussian Negative Log-Likelihood loss for continuous features
                # NLL = 0.5 * [log(var) + (target - mean)^2 / var]
                # More conservative clamping to prevent numerical instability
                logvar_clamped = torch.clamp(logvar[j, :L, :], min=-5, max=2)
                var = torch.exp(logvar_clamped) + 1e-6  # Add epsilon to prevent division by zero
                
                # Compute squared error, replacing NaN with 0
                squared_error = (target[:L, cont_idx] - mean[j, :L, :]) ** 2
                squared_error = torch.where(valid_mask, squared_error, torch.zeros_like(squared_error))
                
                nll = 0.5 * (logvar_clamped + squared_error / var)
                nll_loss = torch.where(valid_mask, nll, torch.zeros_like(nll)).sum()
                
                # KL divergence penalty: penalize predicted variance deviating from empirical
                # 0.5 * [-logvar_pred + log(σ²_emp) + exp(logvar_pred)/σ²_emp - 1]
                safe_emp_var = torch.clamp(empirical_var, min=1e-4)  # Ensure never too small
                kl_div = 0.5 * (-logvar_clamped + torch.log(safe_emp_var) + var / safe_emp_var - 1.0)
                # Apply feature-specific weights and mask NaN positions
                kl_loss = torch.where(valid_mask, kl_div * kl_feature_weights, torch.zeros_like(kl_div)).sum()
                
                # Compute binary cross-entropy loss on flag logits for classification
                bce = F.binary_cross_entropy_with_logits(logits[j, :L, :], target[:L, flag_idx].float(), reduction="sum")
                
                loss_sum += nll_loss
                bce_sum += bce + kl_weight * kl_loss  # Use annealed kl_weight
                count += valid_mask.sum().item()  # Count only valid positions
            
            # Skip batch if no valid data
            if count == 0:
                print(f"  WARNING: Batch {i} has no valid data, skipping")
                continue
                
            loss = (loss_sum + bce_sum) / count
            
            # Check for NaN/Inf in loss before backward
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  WARNING: NaN/Inf loss detected in batch {i}, skipping")
                continue

        scaler.scale(loss).backward()
        
        # Unscale gradients before clipping
        scaler.unscale_(optimizer)
        
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Check for NaN/Inf gradients
        valid_gradients = True
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"  WARNING: NaN/Inf gradient detected in {name}")
                    valid_gradients = False
                    #invaid gradients, then exit
                    sys.exit()
        
        # Only step optimizer if gradients are valid
        if valid_gradients:
            scaler.step(optimizer)
        else:
            print("  Skipping optimizer step due to invalid gradients")
            sys.exit()
        
        scaler.update()
        scheduler.step(epoch + (i + 1) / max(1, len(train_loader)))

        train_loss_sum += loss.item()
        step += 1
        
        # Debug: Check gradients on first batch
        if i == 0:
            grad_norms = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norms.append((name, param.grad.norm().item()))
            if grad_norms:
                print(f"  Sample gradient norms: {grad_norms[:3]}")  # Print first 3

    avg_train = train_loss_sum / max(1, len(train_loader))

    # Validation
    model.eval()
    val_loss_sum, val_bce_sum, val_kl_sum, val_mae_sum, val_count = 0.0, 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids    = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_m  = batch["attention_mask"].to(DEVICE, non_blocking=True)
            targets      = [t.to(DEVICE, non_blocking=True) for t in batch["target"]]

            with amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                # Unpack mean, logvar, and flag outputs from model
                mean, logvar, logits = model(input_ids, attention_m)
                for j, target in enumerate(targets):
                    L = min(mean.shape[1], target.shape[0])
                    
                    # Create mask for valid (non-NaN) continuous features
                    valid_mask = ~torch.isnan(target[:L, cont_idx])  # [L, 3]
                    
                    # Compute NLL loss on continuous features
                    logvar_clamped = torch.clamp(logvar[j, :L, :], min=-10, max=10)
                    var = torch.exp(logvar_clamped) + 1e-6
                    
                    squared_error = (target[:L, cont_idx] - mean[j, :L, :]) ** 2
                    squared_error = torch.where(valid_mask, squared_error, torch.zeros_like(squared_error))
                    
                    nll = 0.5 * (logvar_clamped + squared_error / var)
                    nll_loss = torch.where(valid_mask, nll, torch.zeros_like(nll)).sum()
                    
                    # Compute KL divergence penalty with feature-specific weights
                    safe_emp_var = empirical_var + 1e-6
                    kl_div = 0.5 * (-logvar_clamped + torch.log(safe_emp_var) + var / safe_emp_var - 1.0)
                    # Apply feature-specific weights and mask NaN positions
                    kl_loss = torch.where(valid_mask, kl_div * kl_feature_weights, torch.zeros_like(kl_div)).sum()
                    
                    # Compute BCE loss on binary flag logits
                    bce = F.binary_cross_entropy_with_logits(logits[j, :L, :], target[:L, flag_idx].float(), reduction="sum")
                    # Track MAE on mean predictions for interpretability (only on valid positions)
                    ae = torch.where(valid_mask, (mean[j, :L, :] - target[:L, cont_idx]).abs(), torch.zeros_like(nll))
                    
                    val_loss_sum += nll_loss.item()
                    val_bce_sum  += bce.item()
                    val_kl_sum   += kl_loss.item()
                    val_mae_sum  += ae.sum().item()
                    val_count    += valid_mask.sum().item()  # Count only valid positions

    # Combine NLL, BCE, and KL losses for total validation loss (matches training computation)
    val_loss = (val_loss_sum + val_bce_sum + kl_weight * val_kl_sum) / max(1, val_count)
    val_mae  = val_mae_sum  / max(1, val_count)
    val_kl   = val_kl_sum   / max(1, val_count)
    emp_var_str = f"[{empirical_var[0]:.3f}, {empirical_var[1]:.3f}, {empirical_var[2]:.3f}]"
    print(f"Epoch {epoch+1}: Train={avg_train:.5f}  Val={val_loss:.5f}  MAE={val_mae:.5f}  KL={val_kl:.5f}  EmpiricalVar={emp_var_str}")

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

print("Training complete!")
