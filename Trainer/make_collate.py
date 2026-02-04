import torch
import torch.nn.functional as F
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