"""
PRENA Eval — perplexity on in-domain test + OOD test.
Run after training on Kaggle, with downloaded best.pt.
"""

import json
import math
import torch
from pathlib import Path
from tokenizers import Tokenizer

from model import PrenaNano, PrenaNanoConfig
from data import load_data, make_batches

TOKENIZER_PATH = "../model/prena_tokenizer/tokenizer.json"
CHECKPOINT     = "../model/prena-nano/best.pt"
TEST_FILE      = "../dataset/final/test.jsonl"
TEST_OOD_FILE  = "../dataset/final/test_ood.jsonl"
OUT_FILE       = "../model/prena-nano/eval_results.json"

BLOCK_SIZE = 512
BATCH_SIZE = 8
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def eval_perplexity(model, data, device):
    model.eval()
    total, n = 0.0, 0
    for ids, lbl in make_batches(data, BATCH_SIZE, shuffle=False):
        ids, lbl = ids.to(device), lbl.to(device)
        _, loss = model(ids, lbl)
        total += loss.item()
        n += 1
    avg_loss = total / n
    return avg_loss, math.exp(avg_loss) if avg_loss < 20 else float("inf")


def main():
    print(f"Device: {DEVICE}", flush=True)

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)

    cfg_raw = ckpt["cfg"]
    cfg = PrenaNanoConfig(**cfg_raw) if isinstance(cfg_raw, dict) else cfg_raw
    model = PrenaNano(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint: epoch={ckpt['epoch']}, best_val={ckpt['val_loss']:.4f}", flush=True)

    print("\nEvaluating...", flush=True)

    test_data     = load_data(TEST_FILE,     tokenizer, BLOCK_SIZE)
    test_ood_data = load_data(TEST_OOD_FILE, tokenizer, BLOCK_SIZE)

    loss_id,  ppl_id  = eval_perplexity(model, test_data,     DEVICE)
    loss_ood, ppl_ood = eval_perplexity(model, test_ood_data, DEVICE)

    print(f"\nResults:")
    print(f"  Test (in-domain) : loss={loss_id:.4f}  | ppl={ppl_id:.2f}")
    print(f"  Test (OOD)       : loss={loss_ood:.4f} | ppl={ppl_ood:.2f}")

    results = {
        "checkpoint": CHECKPOINT,
        "epoch": ckpt["epoch"],
        "best_val_loss": ckpt["val_loss"],
        "test_in_domain":  {"loss": loss_id,  "perplexity": ppl_id},
        "test_ood":        {"loss": loss_ood, "perplexity": ppl_ood},
    }

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
