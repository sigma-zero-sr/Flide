"""EmbeddingGemma 2 typed joint decisions: the encoder backbone (backbone.py) and the family (format.py)."""
from .format import EmbeddingGemmaJoint, backbone_dir


def __getattr__(name):
    if name == "EmbeddingGemma2":
        from .backbone import EmbeddingGemma2

        return EmbeddingGemma2
    raise AttributeError(name)


__all__ = ["EmbeddingGemma2", "EmbeddingGemmaJoint", "backbone_dir"]
