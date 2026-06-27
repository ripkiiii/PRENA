"""
PRENA Dataset Processing & Split

Strategy:
- HIGH-PURITY CORE (train/val/test in-domain):
    Synthetic v1 (200) + Synthetic v2 (198) + Antara real (243) = 641 pairs
    Split: 80% train / 10% val / 10% test

- OOD TEST SET (cross-domain generalization):
    Bisnis.com (34) + Liputan6 (73) = 107 pairs
    Kept separate, used only for evaluation in paper

Output:
    dataset/final/train.jsonl        ~513 pairs
    dataset/final/val.jsonl           ~64 pairs
    dataset/final/test.jsonl          ~64 pairs
    dataset/final/test_ood.jsonl      107 pairs
    dataset/final/stats.json          summary
"""

import json
import random
from pathlib import Path
from collections import Counter

SEED = 42
random.seed(SEED)

RAW_SYNTHETIC_V1 = "../dataset/raw/synthetic/prena_synthetic.jsonl"
RAW_SYNTHETIC_V2 = "../dataset/raw/synthetic/prena_synthetic_v2.jsonl"
RAW_REAL         = "../dataset/raw/real/prena_real.jsonl"
OUT_DIR          = Path("../dataset/final")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def save_jsonl(data: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data):>4} pairs → {path.name}")


def main():
    synthetic_v1 = load_jsonl(RAW_SYNTHETIC_V1)
    synthetic_v2 = load_jsonl(RAW_SYNTHETIC_V2)
    synthetic    = synthetic_v1 + synthetic_v2
    real_all     = load_jsonl(RAW_REAL)

    # Split real by source
    antara   = [d for d in real_all if d["source"] == "antaranews.com"]
    bisnis   = [d for d in real_all if d["source"] == "bisnis.com"]
    liputan6 = [d for d in real_all if d["source"] == "liputan6.com"]

    print(f"Loaded:")
    print(f"  Synthetic v1 : {len(synthetic_v1)}")
    print(f"  Synthetic v2 : {len(synthetic_v2)}")
    print(f"  Antara       : {len(antara)}")
    print(f"  Bisnis       : {len(bisnis)}")
    print(f"  Liputan6     : {len(liputan6)}")

    # ── Core: Synthetic + Antara ─────────────────────────────────────
    core = synthetic + antara
    random.shuffle(core)

    n      = len(core)
    n_val  = max(1, round(n * 0.10))
    n_test = max(1, round(n * 0.10))
    n_train = n - n_val - n_test

    train = core[:n_train]
    val   = core[n_train : n_train + n_val]
    test  = core[n_train + n_val :]

    # ── OOD: Bisnis + Liputan6 ────────────────────────────────────────
    ood = bisnis + liputan6
    random.shuffle(ood)

    print(f"\nSplit:")
    save_jsonl(train, OUT_DIR / "train.jsonl")
    save_jsonl(val,   OUT_DIR / "val.jsonl")
    save_jsonl(test,  OUT_DIR / "test.jsonl")
    save_jsonl(ood,   OUT_DIR / "test_ood.jsonl")

    # ── Stats ─────────────────────────────────────────────────────────
    def cat_dist(data):
        return dict(Counter(d.get("category", "unknown") for d in data))

    def src_dist(data):
        return dict(Counter(d.get("source", "unknown") for d in data))

    stats = {
        "seed": SEED,
        "total_pairs": len(synthetic_v1) + len(synthetic_v2) + len(real_all),
        "core_pairs": len(core),
        "ood_pairs": len(ood),
        "splits": {
            "train": {"n": len(train), "categories": cat_dist(train), "sources": src_dist(train)},
            "val":   {"n": len(val),   "categories": cat_dist(val),   "sources": src_dist(val)},
            "test":  {"n": len(test),  "categories": cat_dist(test),  "sources": src_dist(test)},
            "test_ood": {"n": len(ood), "categories": cat_dist(ood),  "sources": src_dist(ood)},
        }
    }

    with open(OUT_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\nStats:")
    print(f"  Total dataset : {stats['total_pairs']} pairs")
    print(f"  Core (train/val/test) : {len(core)} pairs")
    print(f"  OOD test : {len(ood)} pairs")
    print(f"\n  Train categories: {cat_dist(train)}")
    print(f"  Val   categories: {cat_dist(val)}")
    print(f"  OOD   sources:    {src_dist(ood)}")
    print(f"\nDone! Files in {OUT_DIR}/")


if __name__ == "__main__":
    main()
