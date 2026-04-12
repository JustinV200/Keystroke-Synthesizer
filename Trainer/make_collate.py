import torch
import torch.nn.functional as F
def make_collate_fn(pad_token_id: int):
    """Create a collate function for batching variable-length keystroke samples.

    Pads token sequences (input_ids, attention_mask) to the longest sequence in
    the batch, pads character-level mappings (token_to_char_idx) similarly, and
    keeps target feature tensors as a variable-length list.

    Args:
        pad_token_id (int): Token ID used for padding input_ids.

    Returns:
        Callable: A collate function suitable for :class:`torch.utils.data.DataLoader`.
    """
    def collate(batch):
        """Collate a list of dataset samples into a padded batch dict."""
        # Token side (pad for transformer)
        input_ids = [b["input_ids"] for b in batch]
        attn_mask = [b["attention_mask"] for b in batch]
        max_t = max(t.shape[0] for t in input_ids)
        input_ids = [F.pad(t, (0, max_t - t.shape[0]), value=pad_token_id) for t in input_ids]
        attn_mask = [F.pad(m, (0, max_t - m.shape[0]), value=0) for m in attn_mask]
        input_ids = torch.stack(input_ids, dim=0)
        attn_mask = torch.stack(attn_mask, dim=0)

        # Character mapping side (pad token_to_char_idx for char-level expansion)
        t2c_maps = [b["token_to_char_idx"] for b in batch]
        if t2c_maps[0] is not None:
            max_c = max(m.shape[0] for m in t2c_maps)
            char_masks = [torch.ones(m.shape[0], dtype=torch.long) for m in t2c_maps]
            t2c_maps = [F.pad(m, (0, max_c - m.shape[0]), value=0) for m in t2c_maps]
            char_masks = [F.pad(m, (0, max_c - m.shape[0]), value=0) for m in char_masks]
            t2c_maps = torch.stack(t2c_maps, dim=0)
            char_masks = torch.stack(char_masks, dim=0)
        else:
            t2c_maps = None
            char_masks = None

        # Target side (keep variable-length sequences)
        targets = [b["target"] for b in batch]

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "token_to_char_idx": t2c_maps,     # [B, max_chars] or None
            "char_mask": char_masks,            # [B, max_chars] or None
            "target": targets,                  # list of [L_i, F]
        }
    return collate