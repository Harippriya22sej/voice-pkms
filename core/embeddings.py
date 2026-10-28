import numpy as np
from sentence_transformers import SentenceTransformer

_model = None


def _get_model():
    global _model
    if _model is None:  # load once, reuse afterwards
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed(text: str) -> np.ndarray:
    return _get_model().encode(text).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def to_blob(vec: np.ndarray) -> bytes:
    """For saving into the SQLite BLOB column."""
    return vec.astype(np.float32).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)