from __future__ import annotations

import numpy as np

from .tokenizer import SimpleTokenizer
from .transformer import TransformerLM
from .configs import get_preset
import argparse


def chat_loop(temperature: float = 1.0, top_k: int | None = None, top_p: float | None = 0.9, preset: str | None = None):
    tk = SimpleTokenizer()
    if preset:
        cfg = get_preset(preset)
        model = TransformerLM(vocab_size=tk.vocab_size, dim=cfg.dim, n_layers=cfg.n_layers, n_heads=cfg.n_heads, ff_hidden=cfg.ff_hidden, max_len=cfg.max_len)
    else:
        model = TransformerLM(vocab_size=tk.vocab_size, dim=128, n_layers=2, n_heads=4, ff_hidden=256, max_len=128)
    print("KroxAI chat — type 'exit' to quit.")
    ctx = []
    while True:
        user = input("You: ").strip()
        if user.lower() in {"exit", "quit"}:
            break
        prompt = f"Q: {user}\nA: "
        ids = tk.encode(prompt, add_bos=True)
        x = np.array([ids], dtype=np.int64)
        y = model.generate(x, max_new_tokens=64, temperature=temperature, top_k=top_k, top_p=top_p)
        resp = tk.decode(y[0, len(ids):].tolist())
        # Cut at newline or EOS-ish token if present
        resp = resp.split("\n")[0]
        print("KroxAI:", resp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", dest="top_k", type=int, default=None)
    ap.add_argument("--top-p", dest="top_p", type=float, default=0.9)
    ap.add_argument("--preset", type=str, default=None, help="Model size preset: tiny|small|base|large")
    args = ap.parse_args()
    chat_loop(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, preset=args.preset)
