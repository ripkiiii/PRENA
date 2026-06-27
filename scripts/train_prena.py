"""
PRENA Nano Model — Fine-tuning IndoGPT on PRENA dataset
Base: cahya/gpt2-small-indonesian-522M
Device: Apple M1 MPS
"""

import json
import math
import time
import sys
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup

def log(*args, **kwargs):
    print(*args, **kwargs, flush=True)

# ── Config ─────────────────────────────────────────────────────────────
BASE_MODEL   = "cahya/gpt2-small-indonesian-522M"
TRAIN_FILE   = "../dataset/final/train.jsonl"
VAL_FILE     = "../dataset/final/val.jsonl"
OUT_DIR      = Path("../model/prena-nano")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LEN      = 512
BATCH_SIZE   = 4
GRAD_ACCUM   = 4          # effective batch = 16
EPOCHS       = 6
LR           = 3e-4
WARMUP_RATIO = 0.1
DEVICE       = "cpu"  # MPS hangs on .to() for this model size; CPU is stable

log(f"Device: {DEVICE}")
log(f"Base model: {BASE_MODEL}")


# ── Dataset ────────────────────────────────────────────────────────────
def format_example(entry: dict, tokenizer) -> str:
    msgs = entry["messages"]
    user = msgs[0]["content"].strip()
    asst = msgs[1]["content"].strip()
    # ChatML-style format
    text = f"<|user|>\n{user}\n<|assistant|>\n{asst}{tokenizer.eos_token}"
    return text


class PRENADataset(Dataset):
    def __init__(self, path: str, tokenizer, max_len: int):
        self.samples = []
        with open(path, encoding="utf-8") as f:
            raw = [json.loads(l) for l in f if l.strip()]

        for i, entry in enumerate(raw):
            text = format_example(entry, tokenizer)
            enc  = tokenizer(
                text,
                max_length=max_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            ids  = enc["input_ids"].squeeze(0)
            mask = enc["attention_mask"].squeeze(0)

            # Labels: mask padding tokens with -100
            labels = ids.clone()
            labels[mask == 0] = -100

            # Only compute loss on assistant response (after <|assistant|>\n)
            asst_token = tokenizer.encode("<|assistant|>", add_special_tokens=False)
            ids_list = ids.tolist()
            for i in range(len(ids_list) - len(asst_token)):
                if ids_list[i:i+len(asst_token)] == asst_token:
                    labels[:i + len(asst_token) + 1] = -100  # mask prompt
                    break

            self.samples.append({"input_ids": ids, "attention_mask": mask, "labels": labels})
            if (i + 1) % 50 == 0 or (i + 1) == len(raw):
                print(f"  Tokenized {i+1}/{len(raw)}", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── Training ───────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels    = batch["labels"].to(device)
            out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            total_loss += out.loss.item()
            n += 1
    return total_loss / n if n > 0 else float("inf")


def main():
    # Load tokenizer + model
    print("\nLoading tokenizer & model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model = model.to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log(f"Model params: {n_params:.1f}M")

    # Datasets
    log("\nPreparing datasets...")
    log("Tokenizing train...")
    train_ds = PRENADataset(TRAIN_FILE, tokenizer, MAX_LEN)
    log("Tokenizing val...")
    val_ds   = PRENADataset(VAL_FILE,   tokenizer, MAX_LEN)
    log(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps   = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps  = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    log(f"\nTraining for {EPOCHS} epochs, {total_steps} optimizer steps")
    log(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM}\n")

    best_val_loss = float("inf")
    train_log = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        t0 = time.time()

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            labels    = batch["labels"].to(DEVICE)

            out  = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            loss = out.loss / GRAD_ACCUM
            loss.backward()
            epoch_loss += out.loss.item()

            if step % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_train = epoch_loss / len(train_loader)
        val_loss  = evaluate(model, val_loader, DEVICE)
        elapsed   = time.time() - t0
        ppl       = math.exp(val_loss) if val_loss < 20 else float("inf")

        log(f"Epoch {epoch}/{EPOCHS} | train_loss={avg_train:.4f} | val_loss={val_loss:.4f} | ppl={ppl:.2f} | {elapsed:.1f}s")
        train_log.append({"epoch": epoch, "train_loss": avg_train, "val_loss": val_loss, "ppl": ppl})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(OUT_DIR / "best")
            tokenizer.save_pretrained(OUT_DIR / "best")
            log(f"  ✓ Saved best model (val_loss={best_val_loss:.4f})")

    # Save final + log
    model.save_pretrained(OUT_DIR / "final")
    tokenizer.save_pretrained(OUT_DIR / "final")
    with open(OUT_DIR / "train_log.json", "w") as f:
        json.dump({"config": {
            "base_model": BASE_MODEL, "epochs": EPOCHS, "lr": LR,
            "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
            "max_len": MAX_LEN, "device": DEVICE
        }, "log": train_log, "best_val_loss": best_val_loss}, f, indent=2)

    log(f"\nDone! Best val_loss={best_val_loss:.4f} | ppl={math.exp(best_val_loss):.2f}")
    log(f"Model saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
