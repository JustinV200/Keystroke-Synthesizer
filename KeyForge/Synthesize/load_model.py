"""Model loading utilities for KeyForge.

Separated from :mod:`synthesize` so the heavy DeBERTa load can be triggered
(and cached) independently of any inference call.
"""
import json
from collections import OrderedDict

import torch
from transformers import AutoTokenizer

from config import BASE_MODEL, CHECKPOINT_PATH, NUM_CONTINUOUS, STATS_PATH
from Synthesize.TextToKeystrokeModelMultiHead import TextToKeystrokeModelMultiHead


def _strip_dataparallel_prefix(state_dict):
    """Remove ``module.`` prefix added by ``nn.DataParallel`` checkpoints."""
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    cleaned = OrderedDict()
    for k, v in state_dict.items():
        cleaned[k[7:] if k.startswith("module.") else k] = v
    return cleaned


def load_model(
    checkpoint_path=CHECKPOINT_PATH,
    base_model=BASE_MODEL,
    stats_path=STATS_PATH,
    device=None,
):
    """Load the tokenizer, model, and standardization stats once.

    Call this at startup and pass the returned dict to
    :func:`Synthesize.synthesize.predict_keystrokes` to avoid re-loading
    DeBERTa (~400 MB) on every invocation.

    Returns:
        dict: Keys ``tokenizer``, ``model``, ``cont_mean``, ``cont_std``,
        ``device``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = TextToKeystrokeModelMultiHead(base_model, NUM_CONTINUOUS).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint = _strip_dataparallel_prefix(checkpoint)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    with open(stats_path, "r") as f:
        stats = json.load(f)

    cont_mean = torch.tensor(stats["mean"], device=device)
    cont_std = torch.tensor(stats["std"], device=device)

    return {
        "tokenizer": tokenizer,
        "model": model,
        "cont_mean": cont_mean,
        "cont_std": cont_std,
        "device": device,
        # Snapshot of the base-model char_embed so we can revert after a user
        # adapter is loaded without reloading the whole DeBERTa bundle.
        "base_char_embed": {k: v.detach().clone() for k, v in model.char_embed.state_dict().items()},
        # Active per-user calibration (applied in standardized space before
        # de-standardization in predict_keystrokes). None = base model.
        "a_mean": None,
        "b_mean": None,
        "user_name": None,
    }


def apply_user_adapter(bundle, adapter_path):
    """Load a per-user adapter (``adapter.pt``) into an existing bundle.

    Mutates ``bundle`` in place: restores ``char_embed`` from the adapter and
    stores the affine calibration params on the bundle so
    :func:`predict_keystrokes` can apply them.
    """
    device = bundle["device"]
    adapter = torch.load(adapter_path, map_location=device)
    bundle["model"].char_embed.load_state_dict(adapter["char_embed"])
    bundle["a_mean"] = adapter["a_mean"].to(device)
    bundle["b_mean"] = adapter["b_mean"].to(device)
    bundle["user_name"] = adapter.get("user_name")
    return bundle


def clear_user_adapter(bundle):
    """Revert the bundle to the base (non-personalized) model."""
    bundle["model"].char_embed.load_state_dict(bundle["base_char_embed"])
    bundle["a_mean"] = None
    bundle["b_mean"] = None
    bundle["user_name"] = None
    return bundle
