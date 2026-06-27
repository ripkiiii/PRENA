# %%
# PRENA Nano — Kaggle Training Notebook (LLaMA-style, from scratch)
# Upload ke Kaggle sebagai Script, attach dataset "prena-assets"
# Expected input dataset structure:
#   /kaggle/input/prena-assets/
#       tokenizer.json
#       train.jsonl
#       val.jsonl

# %%
import subprocess
subprocess.run(["pip", "install", "tokenizers", "-q"])

# %% [markdown]
# ## 1. Setup

# %%
import json, math, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict
from pathlib import Path
from tokenizers import Tokenizer

TOKENIZER_PATH = "/kaggle/input/datasets/ripkii/prena-assets/tokenizer.json"
TRAIN_FILE     = "/kaggle/input/datasets/ripkii/prena-assets/train.jsonl"
VAL_FILE       = "/kaggle/input/datasets/ripkii/prena-assets/val.jsonl"
OUT_DIR        = Path("/kaggle/working/prena-nano")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

BLOCK_SIZE   = 512
BATCH_SIZE   = 16
GRAD_ACCUM   = 2        # effective batch = 32
MAX_EPOCHS   = 20
LR           = 3e-4
WARMUP_STEPS = 50

# %% [markdown]
# ## 2. Model Architecture (LLaMA-style)

# %%
@dataclass
class PrenaNanoConfig:
    vocab_size  : int   = 16_384
    dim         : int   = 256
    n_layers    : int   = 8
    n_heads     : int   = 4
    n_kv_heads  : int   = 2
    ffn_dim     : int   = 768
    max_seq_len : int   = 512
    norm_eps    : float = 1e-5
    rope_theta  : float = 500_000.0
    logit_cap   : float = 30.0

    @property
    def head_dim(self):
        return self.dim // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def precompute_rope(head_dim, seq_len, theta=500_000.0, device="cpu"):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t     = torch.arange(seq_len, device=device)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    B, T, H, D = x.shape
    x1  = x[..., :D // 2]
    x2  = x[..., D // 2:]
    cos = cos[:T].unsqueeze(0).unsqueeze(2)
    sin = sin[:T].unsqueeze(0).unsqueeze(2)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads    = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim   = cfg.head_dim
        self.groups     = cfg.n_heads // cfg.n_kv_heads

        self.wq = nn.Linear(cfg.dim, cfg.n_heads    * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim,    bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads,    self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        k = k.repeat_interleave(self.groups, dim=2)
        v = v.repeat_interleave(self.groups, dim=2)

        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            is_causal=True,
        )
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.up   = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.down = nn.Linear(cfg.ffn_dim, cfg.dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn      = GroupedQueryAttention(cfg)
        self.ffn_norm  = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn       = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class PrenaNano(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg     = cfg
        self.embed   = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers  = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm    = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        self.lm_head.weight = self.embed.weight

        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x    = self.embed(idx)
        cos  = self.rope_cos[:T]
        sin  = self.rope_sin[:T]

        for layer in self.layers:
            x = layer(x, cos, sin)

        logits = self.lm_head(self.norm(x))
        logits = self.cfg.logit_cap * torch.tanh(logits / self.cfg.logit_cap)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters()) / 1e6

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50):
        for _ in range(max_new_tokens):
            idx_c = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_c)
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx

# %% [markdown]
# ## 3. Data Pipeline

# %%
def load_data(path, tokenizer, block_size):
    pad_id  = tokenizer.token_to_id("<|pad|>")
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d    = json.loads(line)
            user = d["messages"][0]["content"].strip()
            asst = d["messages"][1]["content"].strip()
            prompt_ids   = tokenizer.encode(f"<|user|>\n{user}\n<|assistant|>\n").ids
            response_ids = tokenizer.encode(f"{asst}<|eos|>").ids
            ids    = (prompt_ids + response_ids)[:block_size]
            labels = ([-100] * len(prompt_ids) + response_ids)[:block_size]
            pad    = block_size - len(ids)
            ids    += [pad_id] * pad
            labels += [-100]   * pad
            samples.append((
                torch.tensor(ids,    dtype=torch.long),
                torch.tensor(labels, dtype=torch.long),
            ))
    return samples


def make_batches(data, batch_size, shuffle=True):
    if shuffle:
        data = data[:]; random.shuffle(data)
    for i in range(0, len(data), batch_size):
        c = data[i:i + batch_size]
        yield torch.stack([x[0] for x in c]), torch.stack([x[1] for x in c])

# %% [markdown]
# ## 4. Training

# %%
tokenizer  = Tokenizer.from_file(TOKENIZER_PATH)
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab: {vocab_size}")

cfg   = PrenaNanoConfig(vocab_size=vocab_size)
model = PrenaNano(cfg).to(DEVICE)
print(f"Model: {model.n_params():.1f}M params")

train_data = load_data(TRAIN_FILE, tokenizer, BLOCK_SIZE)
val_data   = load_data(VAL_FILE,   tokenizer, BLOCK_SIZE)
print(f"Train: {len(train_data)} | Val: {len(val_data)}")

# %%
steps_per_epoch = math.ceil(len(train_data) / BATCH_SIZE)
total_steps     = (steps_per_epoch // GRAD_ACCUM) * MAX_EPOCHS

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)

def lr_lambda(step):
    if step < WARMUP_STEPS:
        return step / max(1, WARMUP_STEPS)
    p = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    return max(0.1, 0.5 * (1 + math.cos(math.pi * p)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate():
    model.eval()
    total, n = 0.0, 0
    for ids, lbl in make_batches(val_data, BATCH_SIZE, shuffle=False):
        _, loss = model(ids.to(DEVICE), lbl.to(DEVICE))
        total += loss.item(); n += 1
    model.train()
    return total / n

# %%
best_val = float("inf")
log      = []

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    epoch_loss, t0 = 0.0, time.time()
    optimizer.zero_grad()

    for step, (ids, lbl) in enumerate(make_batches(train_data, BATCH_SIZE), 1):
        _, loss = model(ids.to(DEVICE), lbl.to(DEVICE))
        (loss / GRAD_ACCUM).backward()
        epoch_loss += loss.item()
        if step % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

    avg = epoch_loss / steps_per_epoch
    val = evaluate()
    ppl = math.exp(val) if val < 20 else float("inf")
    print(f"Epoch {epoch:02d}/{MAX_EPOCHS} | train={avg:.4f} | val={val:.4f} | ppl={ppl:.1f} | {time.time()-t0:.1f}s")
    log.append({"epoch": epoch, "train_loss": avg, "val_loss": val, "ppl": ppl})

    if val < best_val:
        best_val = val
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "cfg": asdict(cfg), "val_loss": best_val},
                   OUT_DIR / "best.pt")
        print(f"  ✓ best saved")

# %%
torch.save({"epoch": MAX_EPOCHS, "model_state": model.state_dict(), "cfg": asdict(cfg), "val_loss": val},
           OUT_DIR / "final.pt")

with open(OUT_DIR / "train_log.json", "w") as f:
    json.dump({"log": log, "best_val_loss": best_val}, f, indent=2)

print(f"\nDone! best_val={best_val:.4f} | ppl={math.exp(best_val):.1f}")
print(f"Files saved to {OUT_DIR}")

# %% [markdown]
# ## 5. Sample Generation

# %%
ckpt = torch.load(OUT_DIR / "best.pt")
model.load_state_dict(ckpt["model_state"])
model.eval()

test_prompts = [
    "Buatkan siaran pers tentang peluncuran aplikasi mobile banking PT Bank Nusantara.",
    "Tulis caption Instagram untuk kampanye Hari Kemerdekaan brand fashion lokal Nusara.",
    "Buat pernyataan krisis untuk brand makanan yang viralnya negatif di TikTok.",
]

for prompt in test_prompts:
    ids    = tokenizer.encode(f"<|user|>\n{prompt}\n<|assistant|>\n").ids
    idx    = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    out    = model.generate(idx, max_new_tokens=200, temperature=0.8, top_k=40)
    gen_ids = out[0].tolist()[len(ids):]
    eos_id  = tokenizer.token_to_id("<|eos|>")
    if eos_id in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(eos_id)]
    gen = tokenizer.decode(gen_ids)
    print(f"\nPrompt: {prompt}")
    print(f"Output: {gen[:300]}")
    print("-" * 60)
