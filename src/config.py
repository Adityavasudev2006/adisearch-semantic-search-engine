# config.py - Central configuration for Adisearch.


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
# explored all thse values : 0.70, 0.80, 0.85, 0.90, 0.95
# Lower = more cache hits but wrong matches; Higher = fewer hits but precise
CACHE_SIMILARITY_THRESHOLD = 0.85

# API
API_HOST = "0.0.0.0"
API_PORT = 8000