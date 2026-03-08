# embedder.py - Sentence embedding wrapper.

import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL


class Embedder:
    """Singleton-style embedding wrapper to avoid reloading the model."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self._initialized = True
    
    def embed(self, texts: Union[str, List[str]], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:

        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # crucial: unit vectors for cosine sim
            convert_to_numpy=True
        )
        return embeddings
    
    def embed_single(self, text: str) -> np.ndarray:
        """Convenience method for single query embedding. Returns 1D array."""
        return self.embed([text])[0]


# Module-level singleton
embedder = Embedder()