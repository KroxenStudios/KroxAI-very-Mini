from __future__ import annotations

"""
Lightweight KroxAI shim for local embedding.

Features:
- KroxAI class with handle_user_message(text) -> str
- tries to use HuggingFace transformers if installed (autoregressive model)
- otherwise falls back to a rule-based responder
- repl_main() entrypoint for a simple console REPL (not run automatically)
"""
import os
import time
import random

class KroxAI:
    def __init__(self, agent_name="KroxAI", description=None):
        self.agent_name = agent_name
        self.description = description or "embedded agent"
        self._use_transformers = False
        # lazy import to avoid heavy deps at import time
        try:
            from transformers import pipeline
            model_name = os.environ.get("KROXAI_MODEL", "gpt2")
            # small model by default; users can set env to a local path or HF name
            self.generator = pipeline("text-generation", model=model_name)
            self._use_transformers = True
        except Exception:
            self._use_transformers = False

    def generate(self, text: str, *, temperature: float = 0.9, top_p: float = 0.95,
                 max_new_tokens: int = 160, repetition_penalty: float = 1.1,
                 no_repeat_ngram_size: int = 3) -> str:
        text = (text or "").strip()
        if not text:
            return "Ich habe nichts verstanden. Bitte schreibe eine Frage."
        if not self._use_transformers:
            return self._rules_fallback(text)
        # Clamp values
        temperature = float(max(0.1, min(2.0, temperature)))
        top_p = float(max(0.1, min(1.0, top_p)))
        max_new_tokens = int(max(16, min(512, max_new_tokens)))
        repetition_penalty = float(max(1.0, min(2.0, repetition_penalty)))
        no_repeat_ngram_size = int(max(0, min(6, no_repeat_ngram_size)))
        try:
            out = self.generator(
                text,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                pad_token_id=50256 if "gpt2" in str(getattr(self.generator, 'model', '')) else None,
            )
            if out and isinstance(out, list):
                return out[0]["generated_text"]
            return str(out)
        except TypeError:
            # older transformers w/o max_new_tokens: fallback to max_length
            try:
                out = self.generator(
                    text,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    max_length=len(text) + max_new_tokens,
                    num_return_sequences=1,
                )
                if out and isinstance(out, list):
                    return out[0]["generated_text"]
                return str(out)
            except Exception as e:
                return f"Fehler bei Modellgenerierung: {e}"
        except Exception as e:
            return f"Fehler bei Modellgenerierung: {e}"

    def _rules_fallback(self, text: str) -> str:
        # simple rules fallback
        if any(g in text.lower() for g in ("hallo", "hi", "guten")):
            return f"Hallo! Ich bin {self.agent_name}. Wie kann ich helfen?"
        if "hilfe" in text.lower():
            return "Sag mir kurz, wobei ich helfen soll: Informationen suchen, Dateiaktion vorschlagen oder Konversation führen."
        if text.lower().startswith("was ist ") or text.lower().startswith("was bedeutet "):
            topic = text.split(maxsplit=2)[-1]
            return f"{topic}: Hier ist eine kurze Zusammenfassung (fallback)."
        return f"(Fallback) Ich habe deine Nachricht erhalten: '{text[:120]}'. Kannst du das genauer spezifizieren?"

    def handle_user_message(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return "Ich habe nichts verstanden. Bitte schreibe eine Frage."
        return self.generate(text)


def repl_main():
    print("KroxAI REPL. Type 'quit' to exit.")
    agent = KroxAI()
    while True:
        try:
            txt = input("You: ")
        except EOFError:
            break
        if not txt:
            continue
        if txt.strip().lower() in ("quit", "exit"):
            break
        print("Agent:", agent.handle_user_message(txt))


# Torch CLI runner (only imports heavy deps if executed directly)
if __name__ == "__main__":
    import argparse
    import torch

    # import tokenizer/model utilities with robust fallbacks
    try:
        from .tokenizer import SimpleTokenizer  # type: ignore
        from .hf_tokenizer import HFTokenizer  # type: ignore
        from .torch_model import TorchTransformerLM  # type: ignore
        from .configs import get_preset  # type: ignore
    except Exception:
        try:
            from kroxai.tokenizer import SimpleTokenizer  # type: ignore
            from kroxai.hf_tokenizer import HFTokenizer  # type: ignore
            from kroxai.torch_model import TorchTransformerLM  # type: ignore
            from kroxai.configs import get_preset  # type: ignore
        except Exception:
            from tokenizer import SimpleTokenizer  # type: ignore
            from hf_tokenizer import HFTokenizer  # type: ignore
            from torch_model import TorchTransformerLM  # type: ignore
            from configs import get_preset  # type: ignore

    def cli_main():
        p = argparse.ArgumentParser()
        p.add_argument("--ckpt", type=str, default=None, help="Path to .pt checkpoint")
        p.add_argument("--preset", type=str, default=None, help="Model size preset: tiny|small|base|large")
        p.add_argument("--dim", type=int, default=256)
        p.add_argument("--layers", type=int, default=4)
        p.add_argument("--heads", type=int, default=4)
        p.add_argument("--ff", type=int, default=1024)
        p.add_argument("--max-len", type=int, default=256)
        p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device selection")
        p.add_argument("--temperature", type=float, default=1.0)
        p.add_argument("--top-k", dest="top_k", type=int, default=None)
        p.add_argument("--top-p", dest="top_p", type=float, default=0.9)
        p.add_argument("--tokenizer", type=str, default=None, choices=["simple","hf"], help="Override tokenizer backend (default: from ckpt or simple)")
        p.add_argument("--tokenizer-path", type=str, default=None, help="Path to HF tokenizer if using --tokenizer hf")
        p.add_argument("--pad-id", type=int, default=None)
        p.add_argument("--bos-id", type=int, default=None)
        p.add_argument("--eos-id", type=int, default=None)
        args = p.parse_args()

        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        # Try to read tokenizer config from checkpoint
        sd = None
        if args.ckpt is not None and os.path.exists(args.ckpt):
            sd = torch.load(args.ckpt, map_location="cpu")
        tk = None
        if isinstance(sd, dict) and isinstance(sd.get("config"), dict):
            tconf = sd["config"].get("tokenizer")
            if isinstance(tconf, dict) and tconf.get("type") == "hf" and tconf.get("path") and os.path.exists(tconf["path"]):
                tk = HFTokenizer(tconf["path"])
                for k in ("pad_id","bos_id","eos_id"):
                    if tconf.get(k) is not None:
                        setattr(tk, k.split('_')[0].upper(), int(tconf[k]))
        # Apply CLI override if provided
        if tk is None:
            if args.tokenizer == "hf" and args.tokenizer_path:
                tk = HFTokenizer(args.tokenizer_path)
                for name, val in (("PAD", args.pad_id), ("BOS", args.bos_id), ("EOS", args.eos_id)):
                    if val is not None:
                        setattr(tk, name, int(val))
            else:
                tk = SimpleTokenizer()
        if args.preset:
            cfg = get_preset(args.preset)
            dim, layers, heads, ff, max_len = cfg.dim, cfg.n_layers, cfg.n_heads, cfg.ff_hidden, cfg.max_len
        else:
            dim, layers, heads, ff, max_len = args.dim, args.layers, args.heads, args.ff, args.max_len
        # build model (possibly using ckpt config to override hyperparams)
        if isinstance(sd, dict) and isinstance(sd.get("config"), dict):
            c = sd["config"]
            dim = c.get("dim", dim)
            layers = c.get("n_layers", layers)
            heads = c.get("n_heads", heads)
            ff = c.get("ff_hidden", ff)
            max_len = c.get("max_len", max_len)
        model = TorchTransformerLM(vocab_size=tk.vocab_size, dim=dim, n_layers=layers, n_heads=heads, ff_hidden=ff, max_len=max_len).to(device)
        if args.ckpt is not None:
            if isinstance(sd, dict) and "model" in sd:
                model.load_state_dict(sd["model"])
            else:
                model.load_state_dict(sd)

        model.eval()
        print("KroxAI (Torch) chat — type 'exit' to quit.")
        while True:
            user = input("You: ").strip()
            if user.lower() in {"exit", "quit"}:
                break
            prompt = f"Q: {user}\nA: "
            ids = tk.encode(prompt, add_bos=True)
            x = torch.tensor([ids], dtype=torch.long, device=device)
            gen_kwargs = dict(max_new_tokens=64, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    y = model.generate(x, **gen_kwargs)
            else:
                y = model.generate(x, **gen_kwargs)
            resp = tk.decode(y[0, len(ids):].tolist())
            resp = resp.split("\n")[0]
            print("KroxAI:", resp)

    cli_main()
