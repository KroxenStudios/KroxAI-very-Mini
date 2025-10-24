from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (T, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B,H,T,Hd)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,H,T,T)
        # causal mask
        causal = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(causal, float('-inf'))
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask  # broadcastable

        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, v)  # (B,H,T,Hd)
        context = context.transpose(1, 2).contiguous().view(B, T, C)
        out = self.o_proj(context)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ff_hidden: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, n_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ff_hidden, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x


class TorchTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 256, n_layers: int = 4, n_heads: int = 4, ff_hidden: int = 1024, max_len: int = 512, dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_len = max_len

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_enc = PositionalEncoding(dim, max_len=max_len)
        self.blocks = nn.ModuleList([TransformerBlock(dim, n_heads, ff_hidden, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, T) Long
        B, T = token_ids.shape
        if T > self.max_len:
            token_ids = token_ids[:, -self.max_len:]
            T = token_ids.shape[1]
        x = self.tok_emb(token_ids)  # (B,T,C)
        x = self.pos_enc(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits  # (B,T,V)

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        *,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        no_repeat_ngram_size: Optional[int] = None,
        eos_id: Optional[int] = None,
        min_p: Optional[float] = None,
        typical_p: Optional[float] = None,
        min_new_tokens: int = 0,
    ) -> torch.Tensor:
        """Autoregressive generation with optional penalties and filters.

        Arguments mirror common decoding params. Penalties are applied to logits before sampling:
        - presence_penalty: subtract fixed penalty from tokens that have appeared
        - frequency_penalty: subtract count * penalty for appeared tokens
        - no_repeat_ngram_size: forbid creating any previously seen n-gram of this size
        - eos_id: if produced, stop early (EOS not appended to output)
        - min_p: after softmax, zero out tokens with probability below this absolute threshold
        """
        self.eval()
        out = token_ids
        vocab_size = None

        def _apply_penalties(logits: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
            nonlocal vocab_size
            if vocab_size is None:
                vocab_size = logits.size(-1)
            B = prev.size(0)
            # presence & frequency penalties
            if presence_penalty > 0.0 or frequency_penalty > 0.0:
                for i in range(B):
                    hist = prev[i]
                    # bincount on CPU or GPU depending on device
                    counts = torch.bincount(hist, minlength=vocab_size)
                    if presence_penalty > 0.0:
                        logits[i] = logits[i] - presence_penalty * (counts > 0).to(logits.dtype)
                    if frequency_penalty > 0.0:
                        logits[i] = logits[i] - frequency_penalty * counts.to(logits.dtype)
            # no-repeat n-gram blocking
            if no_repeat_ngram_size and no_repeat_ngram_size > 1:
                n = int(no_repeat_ngram_size)
                for i in range(prev.size(0)):
                    seq = prev[i].tolist()
                    if len(seq) < n:
                        continue
                    prefix = seq[-(n - 1):]
                    banned: set[int] = set()
                    # collect all tokens that would complete a seen n-gram with this prefix
                    for j in range(len(seq) - n + 1):
                        if seq[j:j + n - 1] == prefix:
                            banned.add(seq[j + n - 1])
                    if banned:
                        idx = torch.tensor(list(banned), device=logits.device, dtype=torch.long)
                        logits[i].index_fill_(0, idx, float('-inf'))
            return logits

        produced = 0
        for _ in range(max_new_tokens):
            window = out[:, -self.max_len:]
            logits = self.forward(window)[:, -1, :]
            logits = logits / max(1e-6, temperature)
            logits = _apply_penalties(logits, out)
            # prevent early EOS until min_new_tokens is reached
            if eos_id is not None and produced < int(min_new_tokens):
                try:
                    logits[:, eos_id] = float('-inf')
                except Exception:
                    pass
            if top_k is not None and 0 < top_k < logits.size(-1):
                kth = torch.topk(logits, top_k, dim=-1).values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < kth, torch.full_like(logits, -1e9), logits)
            probs = F.softmax(logits, dim=-1)
            # typical sampling
            if typical_p is not None and 0.0 < typical_p < 1.0:
                next_tok = []
                for i in range(probs.size(0)):
                    p = probs[i]
                    nll = -torch.log(torch.clamp(p, min=1e-20))
                    H = torch.sum(p * nll)
                    dev = torch.abs(nll - H)
                    vals, idx = torch.sort(dev, descending=False)
                    # cumulative mass in order of typicality
                    p_sorted = p[idx]
                    cum = torch.cumsum(p_sorted, dim=-1)
                    mask = cum > typical_p
                    # always keep the first token
                    if mask.numel() > 0:
                        mask[0] = False
                    filt = torch.where(mask, torch.zeros_like(p_sorted), p_sorted)
                    s = filt.sum()
                    if s <= 0:
                        filt = torch.ones_like(p_sorted) / p_sorted.numel()
                    else:
                        filt = filt / s
                    choice = torch.multinomial(filt, num_samples=1)
                    tok = idx[choice]
                    next_tok.append(tok)
                next_tok = torch.stack(next_tok, dim=0).view(-1, 1)
            else:
            # min-p absolute thresholding
                if min_p is not None and 0.0 < min_p < 1.0:
                    probs = torch.where(probs >= min_p, probs, torch.zeros_like(probs))
                    denom = probs.sum(dim=-1, keepdim=True)
                    probs = torch.where(denom > 0, probs / denom, torch.full_like(probs, 1.0 / probs.size(-1)))
                if top_p is not None and 0.0 < top_p < 1.0:
                # nucleus filtering per batch item
                    next_tok = []
                    for i in range(probs.size(0)):
                        p = probs[i]
                        vals, idx = torch.sort(p, descending=True)
                        cum = torch.cumsum(vals, dim=-1)
                        mask = cum > top_p
                        if mask.numel() > 0:
                            mask[0] = False
                        vals = torch.where(mask, torch.zeros_like(vals), vals)
                        s = vals.sum()
                        if s <= 0:
                            vals = torch.ones_like(vals) / vals.numel()
                        else:
                            vals = vals / s
                        choice = torch.multinomial(vals, num_samples=1)
                        next_tok.append(idx[choice])
                    next_tok = torch.stack(next_tok, dim=0).view(-1, 1)
                else:
                    next_tok = torch.multinomial(probs, num_samples=1)

            # early stop on eos
            if eos_id is not None:
                # append then truncate up to (but excluding) eos
                out = torch.cat([out, next_tok], dim=1)
                produced += 1
                # check any eos in batch; for simplicity, if all have eos, break
                if (out[:, -1] == eos_id).all():
                    # remove eos tokens at the end
                    mask = out[:, -1] != eos_id
                    if mask.any():
                        out = out[mask]
                    break
                continue

            out = torch.cat([out, next_tok], dim=1)
            produced += 1
        return out
