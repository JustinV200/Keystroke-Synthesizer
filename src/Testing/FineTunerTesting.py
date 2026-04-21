# quick eval script — drop in src/Testing/eval_adapter.py
import os, sys, shutil, torch, pandas as pd

# Make `src/` importable so `from Trainer...`, `from FineTuner...`, etc. resolve
# regardless of the cwd this script is launched from.
SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
REPO_ROOT = os.path.dirname(SRC_ROOT)

from transformers import AutoTokenizer
from Trainer.TextToKeystrokeModelMultiHead import TextToKeystrokeModelMultiHead
from FineTuner.FineTuner import FineTuner
from dataPipeline.dataLoader import dataLoader
from Trainer.make_collate import make_collate_fn
from Trainer.HeteroscedasticKLLoss import HeteroscedasticKLLoss
from torch.utils.data import DataLoader

USER = "JustinV"
SRC  = os.path.join(SRC_ROOT, "FineTuner", "users", USER)
# (manually copy 70% of capture_*.csv/.txt into a train/ subdir with texts/+csv/
#  and 30% into a test/ subdir before running)

tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
model = TextToKeystrokeModelMultiHead("microsoft/deberta-v3-base", 3)
ckpt = torch.load(os.path.join(REPO_ROOT, "checkpoints", "best_model.pt"), map_location="cpu")
state = {k.removeprefix("module."): v for k, v in ckpt.get("model_state_dict", ckpt).items()}
model.load_state_dict(state, strict=False)

#  baseline MAE on test set (withoug individual adapter)
def mae(model, a_mean, b_mean, test_dir):
    ds = dataLoader(base_dir=test_dir, tokenizer=tok)
    dl = DataLoader(ds, batch_size=4, collate_fn=make_collate_fn(tok.pad_token_id))
    loss = HeteroscedasticKLLoss(torch.ones(3), torch.ones(3), [0,1,2], "cpu")
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for b in dl:
            m, lv = model(b["input_ids"], b["attention_mask"],
                          token_to_char_idx=b["token_to_char_idx"], char_ids=b["char_ids"])
            if a_mean is not None: m = a_mean * m + b_mean
            tot += loss.compute_mae(m, b["target"]); n += 1
    return tot / max(1, n)

print("baseline MAE:", mae(model, None, None, f"{SRC}/test"))

ft = FineTuner(model, USER, tok); ft.user_dir = f"{SRC}/train"; ft._load_data()
ft.fit()
print("finetuned MAE:", mae(model, ft.a_mean, ft.b_mean, f"{SRC}/test"))