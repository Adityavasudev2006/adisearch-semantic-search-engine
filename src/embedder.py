"""
embedder.py - Sentence embedding wrapper.

Why all-MiniLM-L6-v2:
- 384 dimensions: compact enough for fast cosine similarity at scale
- Trained with contrastive learning on 1B+ sentence pairs
- 5x faster than larger models (e.g., all-mpnet-base-v2) with ~5% quality loss
- This tradeoff is correct for a search cache where speed matters

For a production upgrade: switch to 'BAAI/bge-base-en-v1.5' which has
slightly better retrieval benchmarks at the same size.
"""

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
        """
        Embed one or more texts.
        Returns numpy array of shape (n_texts, embedding_dim).
        Always L2-normalized — this makes cosine similarity == dot product,
        which is faster and what ChromaDB's 'cosine' space uses.
        """
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