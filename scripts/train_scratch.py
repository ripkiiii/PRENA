"""
PRENA Nano — Train from scratch
Works locally (smoke test, 2 epochs) and on Kaggle (full run, 20 epochs).
"""

import json
import math
import time
import torch
from dataclasses import asdict
from pathlib import Path
from tokenizers import Tokenizer

from model import PrenaNano, PrenaNanoConfig
from data import load_data, make_batches

# ── Paths (auto-detect Kaggle vs local) ───────────────────────────────
IS_KAGGLE = Path("/kaggle").exists()

if IS_KAGGLE:
    TOKENIZER_PATH = "/kaggle/input/prena-assets/tokenizer.json"
    TRAIN_FILE     = "/kaggle/input/prena-assets/train.jsonl"
    VAL_FILE       = "/kaggle/input/prena-assets/val.jsonl"
    OUT_DIR        = Path("/kaggle/working/prena-nano")
else:
    TOKENIZER_PATH = "../model/prena_tokenizer/tokenizer.json"
    TRAIN_FILE     = "../dataset/final/train.jsonl"
    VAL_FILE       = "../dataset/final/val.jsonl"
    OUT_DIR        = Path("../model/prena-nano-scratch")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparams ────────────────────────────────────────────────────────
BLOCK_SIZE   = 512
BATCH_SIZE   = 8
GRAD_ACCUM   = 4        # effective batch = 32
MAX_EPOCHS   = 20       # full run on Kaggle
LR           = 3e-4
WARMUP_STEPS = 50
SMOKE_TEST   = not IS_KAGGLE  # local = 2 epochs only

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Eval ───────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, val_data, device):
    model.eval()
    total, n = 0.0, 0
    for ids, lbl in make_batches(val_data, BATCH_SIZE, shuffle=False):
        ids, lbl = ids.to(device), lbl.to(device)
        _, loss = model(ids, lbl)
        total += loss.item()
        n += 1
    model.train()
    return total / n if n > 0 else float("inf")


# ── Main ───────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}", flush=True)
    print(f"Mode: {'SMOKE TEST (2 epochs)' if SMOKE_TEST else f'FULL ({MAX_EPOCHS} epochs)'}", flush=True)

    tokenizer  = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size}", flush=True)

    cfg   = PrenaNanoConfig(vocab_size=vocab_size)
    model = PrenaNano(cfg).to(DEVICE)
    print(f"Model: {model.n_params():.1f}M params", flush=True)

    print("Tokenizing...", flush=True)
    train_data = load_data(TRAIN_FILE, tokenizer, BLOCK_SIZE)
    val_data   = load_data(VAL_FILE,   tokenizer, BLOCK_SIZE)
    print(f"Train: {len(train_data)} | Val: {len(val_data)}", flush=True)

    n_epochs        = 2 if SMOKE_TEST else MAX_EPOCHS
    steps_per_epoch = math.ceil(len(train_data) / BATCH_SIZE)
    total_steps     = (steps_per_epoch // GRAD_ACCUM) * n_epochs

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        prog = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\nTraining {n_epochs} epochs | {total_steps} opt steps | eff_batch={BATCH_SIZE * GRAD_ACCUM}\n", flush=True)

    best_val    = float("inf")
    global_step = 0
    log         = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        t0 = time.time()

        for step, (ids, lbl) in enumerate(make_batches(train_data, BATCH_SIZE), 1):
            ids, lbl = ids.to(DEVICE), lbl.to(DEVICE)
            _, loss = model(ids, lbl)
            (loss / GRAD_ACCUM).backward()
            epoch_loss += loss.item()

            if step % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        avg_train = epoch_loss / steps_per_epoch
        val_loss  = evaluate(model, val_data, DEVICE)
        ppl       = math.exp(val_loss) if val_loss < 20 else float("inf")
        elapsed   = time.time() - t0

        print(f"Epoch {epoch:02d}/{n_epochs} | train={avg_train:.4f} | val={val_loss:.4f} | ppl={ppl:.1f} | {elapsed:.1f}s", flush=True)
        log.append({"epoch": epoch, "train_loss": avg_train, "val_loss": val_loss, "ppl": ppl})

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "cfg": cfg, "val_loss": best_val},
                       OUT_DIR / "best.pt")
            print(f"  ✓ best saved (val={best_val:.4f})", flush=True)

    torch.save({"epoch": n_epochs, "model_state": model.state_dict(), "cfg": cfg, "val_loss": val_loss},
               OUT_DIR / "final.pt")

    with open(OUT_DIR / "train_log.json", "w") as f:
        json.dump({"smoke_test": SMOKE_TEST, "cfg": asdict(cfg), "log": log, "best_val_loss": best_val}, f, indent=2)

    print(f"\nDone! best_val={best_val:.4f} | ppl={math.exp(best_val):.1f}", flush=True)
    if SMOKE_TEST:
        print("→ Smoke test passed. Siap naik ke Kaggle.", flush=True)


if __name__ == "__main__":
    main()
