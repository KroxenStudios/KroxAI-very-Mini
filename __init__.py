"""kroxai package public surface.

Keep imports lightweight so importing ``kroxai`` doesn't pull in heavy
optional dependencies (numpy/torch/transformers).

Expose light components (for example ``SimpleTokenizer``) immediately.
Expose heavy components like ``KroxAI`` and ``TransformerLM`` lazily
when first accessed.
"""

from .tokenizer import SimpleTokenizer
from . import data as data_utils

__all__ = ["SimpleTokenizer", "data_utils", "KroxAI", "TransformerLM"]


def _load_kroxai_class():
    """Lazy import of the KroxAI implementation.

    This may raise ImportError if optional runtime dependencies are
    missing (for example torch/transformers). Callers that need the
    full implementation should handle that.
    """
    from .torch_chat import KroxAI

    return KroxAI


def _load_transformer():
    from .transformer import TransformerLM

    return TransformerLM


def __getattr__(name: str):
    # PEP 562 module-level lazy attribute access
    if name == "KroxAI":
        return _load_kroxai_class()
    if name == "TransformerLM":
        return _load_transformer()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + ["KroxAI", "TransformerLM"])
