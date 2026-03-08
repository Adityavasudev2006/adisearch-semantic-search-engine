# scripts/ingest.py - One-time script to embed the corpus and store in ChromaDB.

import sys
import platform

# Only use pysqlite3 on non-Windows systems
if platform.system() != "Windows":
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3

sys.path.insert(0, '.')
import numpy as np
from tqdm import tqdm
from src.data_loader import load_corpus
from src.embedder import embedder
from src.vector_store import vector_store
from src.config import NEWSGROUPS_PATH


def run_ingest():
    print("=" * 60)
    print("ADISEARCH — Corpus Ingestion")
    print("=" * 60)
    
    # Check if already ingested
    existing_count = vector_store.count()
    if existing_count > 0:
        print(f"Vector store already has {existing_count} documents.")
        response = input("Re-ingest? This will reset the collection. [y/N]: ")
        if response.lower() != 'y':
            print("Skipping ingestion.")
            return
        vector_store.reset()
    
    # Load and clean corpus
    print("\n[1/3] Loading corpus...")
    docs = load_corpus()
    print(f"Corpus size: {len(docs)} documents")
    
    # Embed in batches
    print("\n[2/3] Generating embeddings...")
    texts = [d['text'] for d in docs]
    embeddings = embedder.embed(
        texts,
        batch_size=64,
        show_progress=True
    )
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Store in ChromaDB
    print("\n[3/3] Storing in ChromaDB...")
    doc_ids = [d['doc_id'] for d in docs]
    metadatas = [
        {
            "category": d['category'],
            "doc_id": d['doc_id'],
            # cluster fields will be added after clustering step
            "dominant_cluster": -1,
            "cluster_probs": "[]"
        }
        for d in docs
    ]
    
    vector_store.add_documents(
        doc_ids=doc_ids,
        embeddings=embeddings,
        texts=texts,
        metadatas=metadatas
    )
    
    print(f"\n✅ Ingestion complete!")
    print(f"   Documents stored: {vector_store.count()}")
    
    # Save embeddings for clustering step
    np.save("embeddings_cache.npy", embeddings)
    np.save("doc_ids_cache.npy", np.array(doc_ids))
    print("   Embeddings saved to embeddings_cache.npy (for clustering step)")


if __name__ == "__main__":
    run_ingest()