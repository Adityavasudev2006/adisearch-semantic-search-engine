"""
config.py - Central configuration for Adisearch.

Design decisions documented here:
- We use 'all-MiniLM-L6-v2' as our embedding model because:
  * It produces 384-dim embeddings — compact but semantically rich
  * It's fast enough for a 20k-doc corpus on CPU
  * Trained on diverse text including news/forums, matching our domain
  * Outperforms bag-of-words approaches on semantic similarity tasks
  
- We use ChromaDB as our vector store because:
  * It runs fully in-process (no separate server needed)
  * Supports metadata filtering (useful for cluster-aware cache lookup)
  * Persists to disk — embeddings survive restarts
  * Apache 2.0 license, production-ready

- Number of clusters: 15 (not 20, the label count)
  * The 20 newsgroups have known overlaps (e.g., comp.* groups share vocabulary)
  * Empirical testing with silhouette scores + elbow method suggests 12-16 is optimal
  * 15 gives semantically clean clusters while capturing cross-topic docs
  * Justified fully in the notebook with evidence
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
CACHE_DIR = BASE_DIR / "cache_data"

# Dataset — use mini for fast dev, full for final submission
NEWSGROUPS_PATH = DATA_DIR / "20_newsgroups"   # swap to mini_newsgroups for dev
USE_MINI = False  # set True during development

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Vector store
CHROMA_COLLECTION = "newsgroups_corpus"

# Clustering
N_CLUSTERS = 15          # see justification above
FUZZY_M = 2.0            # fuzziness exponent for soft assignments (1=hard, 2=standard fuzzy)

# Semantic cache
# This is THE critical tunable. We explore 0.70, 0.80, 0.85, 0.90, 0.95
# Lower = more cache hits but wrong matches; Higher = fewer hits but precise
# 0.85 is our production default — see analysis in notebook
CACHE_SIMILARITY_THRESHOLD = 0.85

# API
API_HOST = "0.0.0.0"
API_PORT = 8000