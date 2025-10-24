from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional, Iterable, Dict, Any

import numpy as np

from .tokenizer import SimpleTokenizer
from .hf_tokenizer import HFTokenizer


@dataclass
class QAItem:
    q: str
    a: str
    weight: float = 1.0
    persona: Optional[str] = None
    category: Optional[str] = None


def load_qa_json(path: str) -> List[QAItem]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items: List[QAItem] = []
    for obj in data:
        # support optional alias: 'style' -> persona
        persona = obj.get("persona")
        style = obj.get("style")
        if not persona and style:
            persona = style
        items.append(
            QAItem(
                q=obj.get("q", ""),
                a=obj.get("a", ""),
                weight=float(obj.get("weight", 1.0)),
                persona=persona,
                category=obj.get("category"),
            )
        )
    return items


def build_sequences(items: Iterable[QAItem], tokenizer: SimpleTokenizer | HFTokenizer, add_persona_tokens: bool = True) -> List[np.ndarray]:
    """
    Build simple input-target sequences for LM training: "<bos>[persona][category]Q: ...\nA: ...<eos>"
    Returns list of numpy arrays (token ids) for each pair.
    """
    seqs: List[np.ndarray] = []
    for it in items:
        parts: List[str] = []
        parts.append("Q: ")
        parts.append(it.q.strip())
        parts.append("\nA: ")
        parts.append(it.a.strip())
        prompt = "".join(parts)

        prefix = ""
        if add_persona_tokens and it.persona:
            prefix += f"<persona:{it.persona}> "
        if add_persona_tokens and it.category:
            prefix += f"<cat:{it.category}> "

        text = prefix + prompt
        # add BOS/EOS only if tokenizer provides such ids; HF may not have them configured
        add_bos = hasattr(tokenizer, "BOS")
        add_eos = hasattr(tokenizer, "EOS")
        ids = tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
        seqs.append(np.array(ids, dtype=np.int64))
    return seqs


def make_padded_batch(arrays: List[np.ndarray], pad_id: int = 0) -> np.ndarray:
    if not arrays:
        return np.zeros((0, 0), dtype=np.int64)
    T = max(a.shape[0] for a in arrays)
    B = len(arrays)
    out = np.full((B, T), pad_id, dtype=np.int64)
    for i, a in enumerate(arrays):
        out[i, : a.shape[0]] = a
    return out


def iter_minibatches(seqs: List[np.ndarray], batch_size: int) -> Iterable[np.ndarray]:
    for i in range(0, len(seqs), batch_size):
        yield make_padded_batch(seqs[i : i + batch_size])


def save_checkpoint(path: str, params: Dict[str, Any]):
    np.savez(path, **params)


def load_checkpoint(path: str) -> Dict[str, Any]:
    data = np.load(path)
    return {k: data[k] for k in data.files}
