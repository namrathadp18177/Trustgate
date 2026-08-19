"""
Embeddings wrapper. Uses sentence-transformers (all-MiniLM-L6-v2, 384-dim)
locally so the duplication signal works without any API key. This keeps the
"vector storage / similarity search" piece of the stack fully self-contained.
"""
from functools import lru_cache

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedder():
    return _Embedder()


class _Embedder:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def encode(self, text: str):
        if not text or not text.strip():
            return None
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.tolist()
