"""
Credits to https://github.com/karpathy/minGPT
"""

from dataclasses import dataclass
import math
from typing import Optional

from einops import rearrange
import torch
import torch.nn as nn
from torch.nn import functional as F

from .kv_caching import KeysValues, KVCache


@dataclass
class TransformerConfig:
    tokens_per_block: int
    max_blocks: int
    attention: str

    num_layers: int
    num_heads: int
    embed_dim: int

    embed_pdrop: float
    resid_pdrop: float
    attn_pdrop: float

    n_last_frames_drop: int = 0
    last_frames_pdrop: float = 0.0

    @property
    def max_tokens(self):
        return self.tokens_per_block * self.max_blocks


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.drop = nn.Dropout(config.embed_pdrop)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.embed_dim)

    def generate_empty_keys_values(self, n: int, max_tokens: int) -> KeysValues:
        device = self.ln_f.weight.device  # Assumption that all submodules are on the same device
        return KeysValues(n, self.config.num_heads, max_tokens, self.config.embed_dim, self.config.num_layers, device)

    def forward(self, sequences: torch.Tensor, past_keys_values: Optional[KeysValues] = None) -> torch.Tensor:
        assert past_keys_values is None or len(past_keys_values) == len(self.blocks)
        x = self.drop(sequences)
        for i, block in enumerate(self.blocks):
            x = block(x, None if past_keys_values is None else past_keys_values[i])

        x = self.ln_f(x)
        return x


class Block(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.attn = SelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.GELU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x: torch.Tensor, past_keys_values: Optional[KVCache] = None) -> torch.Tensor:
        x_attn = self.attn(self.ln1(x), past_keys_values)
        x = x + x_attn
        x = x + self.mlp(self.ln2(x))
        return x


class MaskedDropout(nn.Module):
    def __init__(self, p=0.1, inplace=True):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x, mask):
        if self.training:
            loc_pool = torch.nonzero(mask)
            loc_samples = loc_pool[torch.rand(loc_pool.size(0)) < self.p]
            if not self.inplace:
                x = x.clone()
            x[..., loc_samples[:, 0], loc_samples[:, 1]] = 0
            frac_affected = loc_pool.size(1) / mask.numel()
            if x.dtype != torch.bool:
                x /= 1 - self.p * frac_affected

        return x


class SelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        assert config.embed_dim % config.num_heads == 0
        assert config.attention in ('causal', 'block_causal')
        self.num_heads = config.num_heads
        self.key = nn.Linear(config.embed_dim, config.embed_dim)
        self.query = nn.Linear(config.embed_dim, config.embed_dim)
        self.value = nn.Linear(config.embed_dim, config.embed_dim)
        self.attn_drop = nn.Dropout(config.attn_pdrop, inplace=False)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)

        causal_mask = torch.tril(torch.ones(config.max_tokens, config.max_tokens))
        block_causal_mask = torch.max(causal_mask, torch.block_diag(
            *[torch.ones(config.tokens_per_block, config.tokens_per_block) for _ in range(config.max_blocks)]))
        self.register_buffer('mask', causal_mask.bool() if config.attention == 'causal' else block_causal_mask.bool())
        # drop obvious previous frames, leave actions
        self.precise_attn_drop = MaskedDropout(config.last_frames_pdrop, inplace=False)
        precise_drop_mask = torch.zeros(config.max_tokens, config.max_tokens)
        for i in range(1, config.max_tokens):  # shift by 1 up to reflect next token prediction
            full_blocks = i // config.tokens_per_block
            for k in range(1, config.n_last_frames_drop + 1):
                if full_blocks > k:
                    precise_drop_mask[i - 1,
                    (full_blocks - k) * config.tokens_per_block:(full_blocks - k + 1) * config.tokens_per_block - 1] = 1

        self.register_buffer("precise_attn_drop_mask", precise_drop_mask.bool())

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> torch.Tensor:
        B, T, C = x.size()
        if kv_cache is not None:
            b, nh, L, c = kv_cache.shape
            assert nh == self.num_heads and b == B and c * nh == C
        else:
            L = 0

        q = self.query(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        k = self.key(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)

        if kv_cache is not None:
            kv_cache.update(k, v)
            k, v = kv_cache.get()

        context_mask = self.mask[L:L + T, :L + T]
        context_mask = self.precise_attn_drop(context_mask, self.precise_attn_drop_mask[L:L + T, :L + T])
        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]):
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=context_mask,
                                                                 dropout_p=self.attn_drop.p if self.training else 0.0)

        y = rearrange(y, 'b h t e -> b t (h e)')
        y = self.resid_drop(self.proj(y))

        return y
