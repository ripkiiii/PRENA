"""
PRENA Nano — LLaMA-style architecture, ~10M params.
Same design as IDK-1 but scaled down:
  IDK-1 : dim=768, 12L, 12Q/4KV heads, ffn=2048 → ~100M
  PRENA  : dim=256,  8L,  4Q/2KV heads, ffn=768  → ~10M
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class PrenaNanoConfig:
    vocab_size  : int   = 16_384
    dim         : int   = 256
    n_layers    : int   = 8
    n_heads     : int   = 4
    n_kv_heads  : int   = 2        # GQA — same pattern as IDK-1
    ffn_dim     : int   = 768      # SwiGLU hidden dim
    max_seq_len : int   = 512
    norm_eps    : float = 1e-5
    rope_theta  : float = 500_000  # LLaMA-3 style, same as IDK-1
    logit_cap   : float = 30.0     # Gemma-2 style soft-capping

    @property
    def head_dim(self):
        return self.dim // self.n_heads


# ── RMSNorm ────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


# ── RoPE ───────────────────────────────────────────────────────────────
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


# ── Grouped Query Attention ────────────────────────────────────────────
class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: PrenaNanoConfig):
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


# ── SwiGLU FFN ────────────────────────────────────────────────────────
class SwiGLU(nn.Module):
    def __init__(self, cfg: PrenaNanoConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.up   = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.down = nn.Linear(cfg.ffn_dim, cfg.dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ── Transformer Block ─────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, cfg: PrenaNanoConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn      = GroupedQueryAttention(cfg)
        self.ffn_norm  = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn       = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ── Full Model ────────────────────────────────────────────────────────
class PrenaNano(nn.Module):
    def __init__(self, cfg: PrenaNanoConfig):
        super().__init__()
        self.cfg    = cfg
        self.embed  = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm   = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        # Weight tying — same as IDK-1
        self.lm_head.weight = self.embed.weight

        # Precompute RoPE
        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x   = self.embed(idx)
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]

        for layer in self.layers:
            x = layer(x, cos, sin)

        logits = self.lm_head(self.norm(x))

        # Logit soft-capping — same as IDK-1
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


DEFAULT_CFG = PrenaNanoConfig()


if __name__ == "__main__":
    cfg   = PrenaNanoConfig()
    model = PrenaNano(cfg)

    total = sum(p.numel() for p in model.parameters())
    print(f"PRENA Nano Config:")
    print(f"  dim={cfg.dim}, layers={cfg.n_layers}, heads={cfg.n_heads}Q/{cfg.n_kv_heads}KV")
    print(f"  ffn_dim={cfg.ffn_dim}, vocab={cfg.vocab_size}, seq={cfg.max_seq_len}")
    print(f"  Total params: {total/1e6:.2f}M")
    print()

    # Verify forward pass
    x = torch.randint(0, cfg.vocab_size, (2, 64))
    logits, loss = model(x, x)
    print(f"  Forward: logits={logits.shape}, loss={loss.item():.4f}")
    print(f"  Logit range: [{logits.min().item():.2f}, {logits.max().item():.2f}] (cap={cfg.logit_cap})")
    assert logits.abs().max().item() <= cfg.logit_cap + 1e-4
    print(f"  Logit capping: OK")
    print()
    print(f"  vs IDK-1: dim=768, 12L, 12Q/4KV, ffn=2048 → ~100M")
    print(f"  PRENA  : dim=256,  8L,  4Q/2KV,  ffn=768  → {total/1e6:.1f}M")
