import os
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataPipeline.dataLoader import dataLoader
from Trainer.make_collate import make_collate_fn
from Trainer.HeteroscedasticKLLoss import HeteroscedasticKLLoss

# Anchor paths to this file so commands work from any cwd.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_DIR = os.path.join(BASE_DIR, "users")
# Global stats the base model was trained with. Reused for each user so the
# fine-tuned outputs stay on the same standardized scale.
GLOBAL_STATS = os.path.join(BASE_DIR, "..", "..", "data", "cont_stats.json")

# Default checkpoint lives at repo_root/checkpoints/best_model.pt.
DEFAULT_CHECKPOINT = os.path.join(BASE_DIR, "..", "..", "checkpoints", "best_model.pt")


class FineTuner:
    # Read an individual's keystroke captures (produced by KeystrokeCapture.py)
    # and fine-tune the base model on them. Only char_embed is unfrozen, plus a
    # small affine calibration head on top of the predicted mean.
    def __init__(self, base_model, user_name, tokenizer, device=None):
        self.base_model = base_model
        self.user_name = user_name
        self.tokenizer = tokenizer
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model.to(self.device)

        self.user_dir = os.path.join(USERS_DIR, user_name)

        # Affine output calibration: y' = a * y + b, applied to predicted mean.
        self.a_mean = nn.Parameter(torch.ones(3, device=self.device))
        self.b_mean = nn.Parameter(torch.zeros(3, device=self.device))

        self._freeze_model()
        self._load_data()

    def _load_data(self, batch_size=8):
        # Reuse the global standardization stats instead of recomputing from a
        # tiny enrollment set — otherwise the user's scale won't match the
        # base model's output scale and the affine head can't correct it.
        user_stats = os.path.join(self.user_dir, "cont_stats.json")
        if not os.path.isfile(user_stats) and os.path.isfile(GLOBAL_STATS):
            os.makedirs(self.user_dir, exist_ok=True)
            shutil.copyfile(GLOBAL_STATS, user_stats)

        ds = dataLoader(base_dir=self.user_dir, tokenizer=self.tokenizer)
        collate_fn = make_collate_fn(self.tokenizer.pad_token_id)
        self.loader = DataLoader(
            ds, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=True, collate_fn=collate_fn,
        )

    def _freeze_model(self):
        for param in self.base_model.parameters():
            param.requires_grad = False
        # Unfreeze char_embed — this is the main biometric signal carrier.
        for param in self.base_model.char_embed.parameters():
            param.requires_grad = True

    def _forward_batch(self, batch):
        """Run the model on a collated batch and return (mean_calibrated, logvar, targets)."""
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        attn_mask = batch["attention_mask"].to(self.device, non_blocking=True)
        t2c = batch.get("token_to_char_idx")
        if t2c is not None:
            t2c = t2c.to(self.device, non_blocking=True)
        char_ids = batch.get("char_ids")
        if char_ids is not None:
            char_ids = char_ids.to(self.device, non_blocking=True)
        targets = [t.to(self.device, non_blocking=True) for t in batch["target"]]

        mean, logvar = self.base_model(input_ids, attn_mask, token_to_char_idx=t2c, char_ids=char_ids)
        # Apply per-user affine calibration on predicted mean. a_mean/b_mean
        # broadcast over [B, T, 3].
        mean = self.a_mean * mean + self.b_mean
        return mean, logvar, targets

    def _empirical_var(self):
        """Compute empirical variance of the 3 continuous features over this user's data."""
        vals = [[], [], []]
        for s in self.loader.dataset.samples:
            feats = torch.as_tensor(s["features"], dtype=torch.float32)
            for i in range(3):
                col = feats[:, i]
                col = col[~torch.isnan(col)]
                if col.numel():
                    vals[i].append(col)
        var = []
        for i in range(3):
            if vals[i]:
                cat = torch.cat(vals[i])
                v = cat.var(unbiased=False).item() if cat.numel() > 1 else 1.0
            else:
                v = 1.0
            var.append(max(v, 1e-3))
        return torch.tensor(var, dtype=torch.float32, device=self.device)

    def fit(self, stage1_epochs=8, stage2_epochs=8,
            lr_affine=1e-2, lr_char_embed=1e-4, weight_decay=1e-2,
            id_reg=1e-2, verbose=True):
        """Two-stage personalization.

        Stage 1: affine-only calibration (fast, ~8 epochs, LR 1e-2).
        Stage 2: affine + char_embed jointly (slow, ~8 epochs, LR 1e-4, WD 1e-2).

        KL weight is held at 0 throughout — we don't need the variance
        regularizer on tiny enrollment data.
        """
        emp_var = self._empirical_var()
        kl_weights = torch.ones(3, device=self.device)
        loss_fn = HeteroscedasticKLLoss(emp_var, kl_weights, cont_idx=[0, 1, 2], device=self.device)

        affine_params = [self.a_mean, self.b_mean]
        char_embed_params = list(self.base_model.char_embed.parameters())

        def run_stage(name, epochs, optim):
            self.base_model.train()
            for epoch in range(epochs):
                loss_sum, n_batches = 0.0, 0
                for batch in self.loader:
                    optim.zero_grad(set_to_none=True)
                    mean, logvar, targets = self._forward_batch(batch)
                    out = loss_fn.forward(mean, logvar, targets, kl_weight=0.0)
                    if out["valid_count"] == 0:
                        continue
                    loss = out["total_loss"]
                    # Identity regularizer: pull affine back toward (a=1, b=0) so a
                    # tiny enrollment set can't push it to a degenerate solution.
                    loss = loss + id_reg * ((self.a_mean - 1.0).pow(2).sum()
                                            + self.b_mean.pow(2).sum())
                    loss.backward()
                    optim.step()
                    loss_sum += loss.item()
                    n_batches += 1
                if verbose:
                    avg = loss_sum / max(1, n_batches)
                    print(f"[{name}] epoch {epoch+1}/{epochs}  loss={avg:.4f}  "
                          f"a={self.a_mean.detach().cpu().tolist()}  "
                          f"b={self.b_mean.detach().cpu().tolist()}")

        # Stage 1: affine only.
        opt1 = torch.optim.AdamW(affine_params, lr=lr_affine)
        run_stage("stage1:affine", stage1_epochs, opt1)

        # Stage 2: affine + char_embed (different LRs per group).
        opt2 = torch.optim.AdamW([
            {"params": affine_params, "lr": lr_affine * 0.1},
            {"params": char_embed_params, "lr": lr_char_embed},
        ], weight_decay=weight_decay)
        run_stage("stage2:affine+char_embed", stage2_epochs, opt2)

    def save(self, path=None):
        """Save the per-user adapter (char_embed + affine). ~32 KB."""
        if path is None:
            os.makedirs(self.user_dir, exist_ok=True)
            path = os.path.join(self.user_dir, "adapter.pt")
        torch.save({
            "char_embed": self.base_model.char_embed.state_dict(),
            "a_mean": self.a_mean.detach().cpu(),
            "b_mean": self.b_mean.detach().cpu(),
            "user_name": self.user_name,
        }, path)
        return path

    def load(self, path=None):
        """Load a previously saved adapter into base_model + self."""
        if path is None:
            path = os.path.join(self.user_dir, "adapter.pt")
        bundle = torch.load(path, map_location=self.device)
        self.base_model.char_embed.load_state_dict(bundle["char_embed"])
        with torch.no_grad():
            self.a_mean.copy_(bundle["a_mean"].to(self.device))
            self.b_mean.copy_(bundle["b_mean"].to(self.device))
        return self
    
#temporary:
if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer
    from Trainer.TextToKeystrokeModelMultiHead import TextToKeystrokeModelMultiHead

    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--base-model", default="microsoft/deberta-v3-base")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--stage1-epochs", type=int, default=8)
    ap.add_argument("--stage2-epochs", type=int, default=8)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = TextToKeystrokeModelMultiHead(args.base_model, num_continuous=3)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    # strip DataParallel 'module.' prefix if present
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)

    ft = FineTuner(model, args.user, tok)
    ft.fit(stage1_epochs=args.stage1_epochs, stage2_epochs=args.stage2_epochs)
    out = ft.save()
    print(f"Saved adapter to {out}")