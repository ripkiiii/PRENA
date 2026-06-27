"""
PRENA Smoke Test — jalanin ini dulu sebelum ke Kaggle.
Verifikasi: tokenizer, model shape, data pipeline, 2 epoch training, sample generation.
"""

import sys
import torch
from pathlib import Path
from tokenizers import Tokenizer

from model import PrenaNano, PrenaNanoConfig
from data import load_data, make_batches

TOKENIZER_PATH = "../model/prena_tokenizer/tokenizer.json"
TRAIN_FILE     = "../dataset/final/train.jsonl"
VAL_FILE       = "../dataset/final/val.jsonl"

BLOCK_SIZE = 512
BATCH_SIZE = 4
DEVICE     = "cpu"

PASS = "✓"
FAIL = "✗"


def check(name, fn):
    try:
        result = fn()
        print(f"  {PASS} {name}: {result}", flush=True)
        return True
    except Exception as e:
        print(f"  {FAIL} {name}: {e}", flush=True)
        return False


def main():
    ok = True
    print("=" * 50, flush=True)
    print("PRENA SMOKE TEST", flush=True)
    print("=" * 50, flush=True)

    # 1. Tokenizer
    print("\n[1] Tokenizer", flush=True)
    if not Path(TOKENIZER_PATH).exists():
        print(f"  {FAIL} tokenizer not found → run: python tokenizer_train.py", flush=True)
        sys.exit(1)

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()
    ok &= check("vocab_size", lambda: f"{vocab_size}")
    ok &= check("special tokens", lambda: f"<|eos|>={tokenizer.token_to_id('<|eos|>')}, <|user|>={tokenizer.token_to_id('<|user|>')}, <|assistant|>={tokenizer.token_to_id('<|assistant|>')}")
    ok &= check("encode/decode", lambda: tokenizer.decode(tokenizer.encode("Siaran pers PT Nusantara").ids))

    # 2. Model
    print("\n[2] Model", flush=True)
    cfg   = PrenaNanoConfig(vocab_size=vocab_size)
    model = PrenaNano(cfg)
    ok &= check("param count", lambda: f"{model.n_params():.1f}M")
    ok &= check("forward pass", lambda: (
        lambda out: f"logits={out[0].shape}, loss={out[1].item():.4f}"
    )(model(torch.randint(0, vocab_size, (2, 64)), torch.randint(0, vocab_size, (2, 64)))))

    # 3. Data pipeline
    print("\n[3] Data pipeline", flush=True)
    train_data = load_data(TRAIN_FILE, tokenizer, BLOCK_SIZE)
    val_data   = load_data(VAL_FILE,   tokenizer, BLOCK_SIZE)
    ok &= check("train samples", lambda: len(train_data))
    ok &= check("val samples",   lambda: len(val_data))
    ids, lbl = next(make_batches(train_data, BATCH_SIZE))
    ok &= check("batch shape",   lambda: f"ids={ids.shape}, labels={lbl.shape}")
    ok &= check("labels masked", lambda: f"{(lbl == -100).float().mean().item():.0%} masked")

    # 4. Mini training (2 epochs)
    print("\n[4] Training (2 epochs)", flush=True)
    model = model.to(DEVICE)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses = []

    for epoch in range(1, 3):
        epoch_loss, n = 0.0, 0
        for ids_b, lbl_b in make_batches(train_data, BATCH_SIZE):
            ids_b, lbl_b = ids_b.to(DEVICE), lbl_b.to(DEVICE)
            _, loss = model(ids_b, lbl_b)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            n += 1
        avg = epoch_loss / n
        losses.append(avg)
        print(f"  Epoch {epoch}: loss={avg:.4f}", flush=True)

    ok &= check("loss decreased", lambda: "yes" if losses[1] < losses[0] else f"NO — {losses[0]:.4f} → {losses[1]:.4f}")

    # 5. Generation
    print("\n[5] Sample generation", flush=True)
    model.eval()
    user_id = tokenizer.token_to_id("<|user|>")
    ast_id  = tokenizer.token_to_id("<|assistant|>")
    eos_id  = tokenizer.token_to_id("<|eos|>")

    prompt = "Buatkan siaran pers peluncuran aplikasi mobile PT Teknologi Nusantara."
    prompt_ids = tokenizer.encode(f"<|user|>\n{prompt}\n<|assistant|>\n").ids
    idx = torch.tensor([prompt_ids], dtype=torch.long)

    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=80, temperature=0.8, top_k=40)

    generated = tokenizer.decode(out[0].tolist()[len(prompt_ids):])
    print(f"  Prompt: {prompt[:60]}...", flush=True)
    print(f"  Output: {generated[:200]}", flush=True)
    ok &= check("generation runs", lambda: f"{len(generated)} chars generated")

    # Summary
    print("\n" + "=" * 50, flush=True)
    if ok:
        print(f"{PASS} SMOKE TEST PASSED — siap naik ke Kaggle!", flush=True)
    else:
        print(f"{FAIL} SMOKE TEST FAILED — fix errors above dulu.", flush=True)
    print("=" * 50, flush=True)


if __name__ == "__main__":
    main()
