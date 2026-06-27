"""
PRENA dataset pipeline.
Handles tokenization and batching for training/eval.
"""

import json
import random
import torch
from tokenizers import Tokenizer


def load_data(path: str, tokenizer: Tokenizer, block_size: int) -> list:
    """
    Load PRENA jsonl, format as ChatML, tokenize.
    Prompt tokens are masked (-100) so loss is only on assistant response.

    Returns list of (input_ids, labels) tensors, each of length block_size.
    """
    pad_id = tokenizer.token_to_id("<|pad|>")

    samples = []
    with open(path, encoding="utf-8") as f:
        entries = [json.loads(l) for l in f if l.strip()]

    for entry in entries:
        user = entry["messages"][0]["content"].strip()
        asst = entry["messages"][1]["content"].strip()

        prompt_ids   = tokenizer.encode(f"<|user|>\n{user}\n<|assistant|>\n").ids
        response_ids = tokenizer.encode(f"{asst}<|eos|>").ids

        ids    = (prompt_ids + response_ids)[:block_size]
        labels = ([-100] * len(prompt_ids) + response_ids)[:block_size]

        pad_len = block_size - len(ids)
        ids    += [pad_id] * pad_len
        labels += [-100]   * pad_len

        samples.append((
            torch.tensor(ids,    dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        ))

    return samples


def make_batches(data: list, batch_size: int, shuffle: bool = True):
    """Yield (input_ids, labels) batches."""
    if shuffle:
        data = data[:]
        random.shuffle(data)
    for i in range(0, len(data), batch_size):
        chunk = data[i:i + batch_size]
        ids = torch.stack([x[0] for x in chunk])
        lbl = torch.stack([x[1] for x in chunk])
        yield ids, lbl
