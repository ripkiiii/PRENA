"""
Train BPE tokenizer on all PRENA data.
Output: ../model/prena_tokenizer/
"""

import json
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing

VOCAB_SIZE = 16384
OUT_DIR    = Path("../model/prena_tokenizer")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILES = [
    "../dataset/final/train.jsonl",
    "../dataset/final/val.jsonl",
    "../dataset/final/test.jsonl",
    "../dataset/final/test_ood.jsonl",
]

SPECIAL_TOKENS = ["<|pad|>", "<|eos|>", "<|user|>", "<|assistant|>"]


def iter_texts():
    for path in DATA_FILES:
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                for msg in d["messages"]:
                    yield msg["content"]


def main():
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=1,
        show_progress=True,
    )

    print(f"Training BPE tokenizer (vocab_size={VOCAB_SIZE})...")
    tokenizer.train_from_iterator(iter_texts(), trainer=trainer)

    tokenizer.save(str(OUT_DIR / "tokenizer.json"))
    print(f"Saved to {OUT_DIR}/tokenizer.json")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    # Quick test
    out = tokenizer.encode("Buatkan siaran pers tentang peluncuran produk terbaru PT Nusantara Digital.")
    print(f"Test encode: {len(out.tokens)} tokens → {out.tokens[:10]}...")


if __name__ == "__main__":
    main()
