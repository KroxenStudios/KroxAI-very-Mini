from __future__ import annotations

from pathlib import Path
from typing import List

class HFTokenizer:
    """
    Thin wrapper around HuggingFace tokenizers or SentencePiece.
    Accepts a path to a tokenizer JSON (tokenizers) or .model (sentencepiece).
    Exposes encode/decode similar to SimpleTokenizer and a vocab_size property.
    """
    def __init__(self, tokenizer_path: str):
        self.path = str(tokenizer_path)
        p = Path(self.path)
        if not p.exists():
            raise FileNotFoundError(f"tokenizer file not found: {self.path}")
        self._impl = None
        self._sp = None
        if p.suffix.lower() == ".json":
            try:
                from tokenizers import Tokenizer
            except Exception as e:
                raise RuntimeError("Install 'tokenizers' to use JSON tokenizers") from e
            self._impl = Tokenizer.from_file(self.path)
            self._vocab_size = self._impl.get_vocab_size()
            # Try to detect common special token ids
            try:
                tok_to_id = getattr(self._impl, 'token_to_id', None)
                if callable(tok_to_id):
                    def _try_set(name: str, candidates: list[str]):
                        for t in candidates:
                            tid = tok_to_id(t)
                            if isinstance(tid, int) and tid >= 0:
                                setattr(self, name, int(tid))
                                return True
                        return False
                    _try_set('PAD', ['<pad>', '[PAD]'])
                    _try_set('BOS', ['<s>', '[CLS]', '<bos>'])
                    _try_set('EOS', ['</s>', '[SEP]', '<eos>'])
            except Exception:
                pass
        elif p.suffix.lower() in (".model", ".spm"):
            try:
                import sentencepiece as spm
            except Exception as e:
                raise RuntimeError("Install 'sentencepiece' to use .model tokenizers") from e
            self._sp = spm.SentencePieceProcessor(model_file=self.path)
            self._vocab_size = int(self._sp.vocab_size())
            # Try to set common special ids if defined
            try:
                if hasattr(self._sp, 'pad_id'):
                    pid = int(self._sp.pad_id())
                    if pid >= 0:
                        setattr(self, 'PAD', pid)
                if hasattr(self._sp, 'bos_id'):
                    bid = int(self._sp.bos_id())
                    if bid >= 0:
                        setattr(self, 'BOS', bid)
                if hasattr(self._sp, 'eos_id'):
                    eid = int(self._sp.eos_id())
                    if eid >= 0:
                        setattr(self, 'EOS', eid)
            except Exception:
                pass
        else:
            raise ValueError(f"Unsupported tokenizer format: {p.suffix}")

    @property
    def vocab_size(self) -> int:
        return int(self._vocab_size)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int]
        if self._impl is not None:
            # HF tokenizers returns ids in .ids
            ids = list(self._impl.encode(text).ids)
        else:
            ids = list(self._sp.encode(text, out_type=int))
        if add_bos and hasattr(self, 'BOS'):
            ids = [getattr(self, 'BOS')] + ids
        if add_eos and hasattr(self, 'EOS'):
            ids = ids + [getattr(self, 'EOS')]
        return ids

    def decode(self, ids: List[int]) -> str:
        if self._impl is not None:
            return self._impl.decode(ids)
        return self._sp.decode(ids)
