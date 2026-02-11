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
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataPipeline.dataLoader import dataLoader
from .TextToKeystrokeModelMultiHead import TextToKeystrokeModelMultiHead
from .make_collate import make_collate_fn
from .config import *  # Import all configuration constants
from .utils import *
from .HeteroscedasticKLLoss import *
# Data loading


class Trainer():


    def __init__(self):

        print("Loading tokenizer and datasets...")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


        self.full_dataset = dataLoader(
            base_dir="data", tokenizer=self.tokenizer,
            preprocess=True, max_length=MAX_TOKENS
        )

        # split into train/val
        self.n_total = len(self.full_dataset)
        self.n_val   = int(0.2 * self.n_total) # 20% for validation
        self.n_train = self.n_total - self.n_val # rest for training
        # random split with fixed seed for reproducibility
        self.train_dataset, self.val_dataset = random_split(
            self.full_dataset, [self.n_train, self.n_val],
            generator=torch.Generator().manual_seed(42)
        )
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        collate_fn = make_collate_fn(pad_id)

        # Data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=0, pin_memory=True, collate_fn=collate_fn)
        
        self.val_loader   = DataLoader(self.val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=0, pin_memory=True, collate_fn=collate_fn)

        print(f"Train samples: {len(self.train_dataset)}, Val samples: {len(self.val_dataset)}")
        # Initialize model, optimizer, scheduler, scaler
        self.model = self._create_model()
        #create seperate parameter groups for logvar head
        logvar_head_params = [p for n, p in self.model.named_parameters() if 'logvar_head' in n]
        other_params = [p for n, p in self.model.named_parameters() if 'logvar_head' not in n]
        
        self.optimizer = torch.optim.AdamW([
            {'params': other_params, 'lr': LR},
            {'params': logvar_head_params, 'lr': LR * 5},  # 5x LR for logvar head
        ], weight_decay=WEIGHT_DECAY)


        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=3, T_mult=2)
        self.scaler    = amp.GradScaler(device="cuda" if DEVICE.type == "cuda" else "cpu")

        #flags
        # - 3 continuous features: DwellTime, FlightTime, typing_speed
        # - 7 binary flags: is_letter, is_digit, is_punct, is_space, is_backspace, is_enter, is_shift

        self.cont_idx = [0, 1, 2]  # DwellTime, FlightTime, typing_speed
        self.flag_idx = [3, 4, 5, 6, 7, 8, 9]  # 7 binary flags
        
        # Compute empirical variance from training data, used for kl weights, train towards realistic uncertainty estimates
        self.empirical_var = self._compute_empirical_variance()
        self.kl_feature_weights = self._get_kl_feature_weights()
        self.bad_batches = 0  # Counter for batches with invalid gradients
        #heteroscedasticKLLoss.py
        self.heteroscedastic_loss = HeteroscedasticKLLoss(self.empirical_var, self.kl_feature_weights, self.cont_idx, self.flag_idx, DEVICE)


    def _compute_empirical_variance(self):
        return compute_empirical_variance(self.train_dataset, DEVICE)
    

    def _get_kl_feature_weights(self):
        return torch.tensor(KL_FEATURE_WEIGHTS, device=DEVICE)  # [3]


    def _get_feature_size(self):
        probe = next(iter(self.train_loader))
        return probe["target"][0].shape[-1]
        

    def _create_model(self):
        model = TextToKeystrokeModelMultiHead(BASE_MODEL, num_continuous=3, num_flags=7).to(DEVICE)
        if torch.cuda.device_count() > 1: # use multiple GPUs if available
            model = nn.DataParallel(model)
        return model
    

    def _check_gradients(self):
        """Check for NaN/Inf gradients and return gradient statistics"""
        grad_stats = {'has_nan': False, 'has_inf': False, 'max_norm': 0.0, 'problem_params': []}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_stats['max_norm'] = max(grad_stats['max_norm'], grad_norm)
                
                if torch.isnan(param.grad).any():
                    grad_stats['has_nan'] = True
                    grad_stats['problem_params'].append(f"{name}(NaN)")
                    
                if torch.isinf(param.grad).any():
                    grad_stats['has_inf'] = True
                    grad_stats['problem_params'].append(f"{name}(Inf)")
                    
                # Check for extremely large gradients
                if grad_norm > 100.0:
                    grad_stats['problem_params'].append(f"{name}(large:{grad_norm:.2f})")
        
        return grad_stats
    
    def _train(self):
        #define variables for training loop
        best_val, patience, step = float("inf"), 0, 0
        for epoch in range(EPOCHS):
            # Compute current KL weight using linear annealing schedule
            if epoch < KL_ANNEAL_EPOCHS:
                kl_weight = KL_WEIGHT_START + (KL_WEIGHT_END - KL_WEIGHT_START) * (epoch / KL_ANNEAL_EPOCHS)
            else:
                kl_weight = KL_WEIGHT_END
            print(f"Epoch {epoch+1}/{EPOCHS} - KL weight: {kl_weight:.5f}")
    
            self.model.train()
            train_loss_sum = 0.0
            for i, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
                input_ids    = batch["input_ids"].to(DEVICE, non_blocking=True)
                attention_m  = batch["attention_mask"].to(DEVICE, non_blocking=True)
                targets      = [t.to(DEVICE, non_blocking=True) for t in batch["target"]]

                self.optimizer.zero_grad(set_to_none=True)
                with amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                    # Model now returns mean, logvar, logits
                    mean, logvar, logits = self.model(input_ids, attention_m)
                    # Debug: Check for NaN in model outputs
                    checkforNans(mean, logvar, logits, i, input_ids, attention_m, targets)
                    
                    # Compute loss using HeteroscedasticKLLoss (same as validation)
                    loss_dict = self.heteroscedastic_loss.forward(mean, logvar, logits, targets, kl_weight)
                    loss = loss_dict['total_loss']
                    
                    # Skip batch if no valid data
                    if loss_dict['valid_count'] == 0:
                        print(f"  WARNING: Batch {i} has no valid data, skipping")
                        continue
                    
                    # Check for NaN/Inf in loss before backward
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"  WARNING: NaN/Inf loss detected in batch {i}, skipping")
                        continue

                self.scaler.scale(loss).backward()
                
                # Unscale gradients before clipping
                self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping
                #changed from 1 to 75, only catch extreme outliers.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                
                # Check for NaN/Inf gradients with detailed monitoring
                grad_stats = self._check_gradients()

                if grad_stats['has_nan'] or grad_stats['has_inf']:
                    print(f"  WARNING: Invalid gradients in batch {i}, skipping update")
                    print(f"    Problem parameters: {grad_stats['problem_params'][:5]}")
                    # Zero out gradients so they don't corrupt model weights
                    self.optimizer.zero_grad(set_to_none=True)
                    # Still update scaler so it reduces its scale factor
                    self.scaler.update()
                    self.bad_batches += 1
                    if self.bad_batches > 20:
                        print("  ERROR: Too many bad batches (>20), stopping training.")
                        return
                    continue
                self.bad_batches = 0
                # Check for suspiciously large gradients
                if grad_stats['max_norm'] > 50.0:
                    print(f"  WARNING: Large gradient detected (norm={grad_stats['max_norm']:.3f})")
                    print(f"    Problem parameters: {grad_stats['problem_params'][:3]}")
                
                # Step optimizer (we've already checked for invalid gradients above)
                self.scaler.step(self.optimizer)
                
                self.scaler.update()
                self.scheduler.step(epoch + (i + 1) / max(1, len(self.train_loader)))

                train_loss_sum += loss.item()
                step += 1
                
                # Debug: Check gradients on first batch
                if i == 0:
                    grad_norms = []
                    logvar_head_weights = None
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            grad_norms.append((name, param.grad.norm().item()))
                            # Track logvar_head weights specifically
                            if 'logvar_head' in name and 'weight' in name:
                                logvar_head_weights = param.data
                    if grad_norms:
                        print(f"  Sample gradient norms: {grad_norms[:3]}")  # Print first 3
                        if logvar_head_weights is not None:
                            print(f"  Logvar head range: [{logvar_head_weights.min():.3f}, {logvar_head_weights.max():.3f}]")

            avg_train = train_loss_sum / max(1, len(self.train_loader))
            
            # Run validation and update early stopping
            val_loss, val_mae = self._validate(epoch, kl_weight, avg_train)
            
            if val_loss < best_val:
                best_val = val_loss
                patience = 0
                torch.save(self.model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))
                print("Saved new best model.")
            else:
                patience += 1
                if patience >= PATIENCE:
                    print("Early stopping triggered.")
                    break
        
        print("Training complete!")

    def _validate(self, epoch, kl_weight, avg_train):
        # Validation
        self.model.eval()
        val_loss_sum, val_mae_sum, val_count = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids    = batch["input_ids"].to(DEVICE, non_blocking=True)
                attention_m  = batch["attention_mask"].to(DEVICE, non_blocking=True)
                targets      = [t.to(DEVICE, non_blocking=True) for t in batch["target"]]

                with amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                    # Unpack mean, logvar, and flag outputs from model
                    mean, logvar, logits = self.model(input_ids, attention_m)
                    
                    # Compute validation loss using HeteroscedasticKLLoss
                    loss_dict = self.heteroscedastic_loss.forward(mean, logvar, logits, targets, kl_weight)
                    
                    # Compute MAE for interpretability
                    mae = self.heteroscedastic_loss.compute_mae(mean, targets)
                    
                    # Accumulate validation metrics
                    if loss_dict['valid_count'] > 0:
                        val_loss_sum += loss_dict['total_loss'].item() * loss_dict['valid_count']
                        val_mae_sum += mae * loss_dict['valid_count']
                        val_count += loss_dict['valid_count']

        # Compute validation metrics
        val_loss = val_loss_sum / max(1, val_count)
        val_mae  = val_mae_sum  / max(1, val_count)
        emp_var_str = f"[{self.empirical_var[0]:.3f}, {self.empirical_var[1]:.3f}, {self.empirical_var[2]:.3f}]"
        print(f"Epoch {epoch+1}: Train={avg_train:.5f}  Val={val_loss:.5f}  MAE={val_mae:.5f}  EmpiricalVar={emp_var_str}")
        
        return val_loss, val_mae

    def train(self):
        """Public method to start training"""
        self._train()
        
    def save_model(self, path=None):
        """Save the trained model"""
        if path is None:
            path = os.path.join(OUTPUT_DIR, "final_model.pt")
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
        
    def load_model(self, path):
        """Load a pre-trained model"""
        self.model.load_state_dict(torch.load(path, map_location=DEVICE))
        self.model.eval()
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    # Example usage
    trainer = Trainer()
    trainer.train()
