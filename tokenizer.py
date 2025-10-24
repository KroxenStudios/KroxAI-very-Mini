from __future__ import annotations

from typing import List


class SimpleTokenizer:
    """
    A very simple byte-level tokenizer with special tokens:
    - 0: <pad>
    - 1: <bos>
    - 2: <eos>
    Byte tokens occupy 3..258 (inclusive) mapping to 0..255.

    This is intentionally simple to keep dependencies minimal and
    to satisfy unit tests relying on existence of a tokenizer.
    """

    PAD = 0
    BOS = 1
    EOS = 2
    BYTE_OFFSET = 3
    BYTE_VOCAB = 256

    def __init__(self):
        self.vocab_size = self.BYTE_OFFSET + self.BYTE_VOCAB

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.BOS)
        for b in text.encode("utf-8", errors="replace"):
            ids.append(self.BYTE_OFFSET + int(b))
        if add_eos:
            ids.append(self.EOS)
        return ids

    def decode(self, ids: List[int]) -> str:
        bytes_out = bytearray()
        for i in ids:
            if i in (self.PAD, self.BOS, self.EOS):
                continue
            b = i - self.BYTE_OFFSET
            if 0 <= b < 256:
                bytes_out.append(b)
        return bytes_out.decode("utf-8", errors="replace")
