"""
PRENA Inference — generate Digital PR content from prompt.
Usage:
    python inference.py --prompt "Buatkan siaran pers peluncuran produk..."
    python inference.py  # interactive mode
"""

import argparse
import torch
from pathlib import Path
from tokenizers import Tokenizer

from model import PrenaNano, PrenaNanoConfig

TOKENIZER_PATH = "../model/prena_tokenizer/tokenizer.json"
CHECKPOINT     = "../model/prena-nano/best.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    ckpt    = torch.load(CHECKPOINT, map_location=DEVICE)
    cfg_raw = ckpt["cfg"]
    cfg     = PrenaNanoConfig(**cfg_raw) if isinstance(cfg_raw, dict) else cfg_raw
    model   = PrenaNano(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"PRENA Nano loaded | epoch={ckpt['epoch']} | val_loss={ckpt['val_loss']:.4f}", flush=True)
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens=300, temperature=0.8, top_k=40, rep_penalty=1.3) -> str:
    eos_id = tokenizer.token_to_id("<|eos|>")

    prompt_ids = tokenizer.encode(f"<|user|>\n{prompt.strip()}\n<|assistant|>\n").ids
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_c = idx[:, -model.cfg.max_seq_len:]
            logits, _ = model(idx_c)
            logits = logits[:, -1, :] / temperature

            # repetition penalty
            if rep_penalty != 1.0:
                for token_id in set(idx[0].tolist()):
                    logits[0, token_id] /= rep_penalty

            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            import torch.nn.functional as F
            next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, next_tok], dim=1)

            if next_tok.item() == eos_id:
                break

    generated_ids = idx[0].tolist()[len(prompt_ids):]
    if eos_id in generated_ids:
        generated_ids = generated_ids[:generated_ids.index(eos_id)]

    return tokenizer.decode(generated_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--rep_penalty", type=float, default=1.3)
    args = parser.parse_args()

    model, tokenizer = load_model()

    if args.prompt:
        out = generate(model, tokenizer, args.prompt, args.max_tokens, args.temperature, args.top_k, args.rep_penalty)
        print(f"\n{'='*60}")
        print(f"PROMPT: {args.prompt}")
        print(f"{'='*60}")
        print(out)
    else:
        # Interactive mode
        print("\nPRENA Nano — Interactive Mode (ketik 'quit' untuk keluar)\n")
        while True:
            prompt = input("Prompt: ").strip()
            if prompt.lower() in ("quit", "exit", "q"):
                break
            if not prompt:
                continue
            out = generate(model, tokenizer, prompt, args.max_tokens, args.temperature, args.top_k, args.rep_penalty)
            print(f"\n{out}\n{'─'*60}\n")


if __name__ == "__main__":
    main()
