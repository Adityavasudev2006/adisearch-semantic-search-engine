# vector_store.py - ChromaDB vector store wrapper.

import json
import numpy as np
from typing import List, Dict, Optional, Any
import chromadb
from chromadb.config import Settings

from src.config import CHROMA_DIR, CHROMA_COLLECTION


class VectorStore:
    """ChromaDB-backed vector store for the newsgroups corpus."""
    
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"}  # cosine similarity for normalized embeddings
        )
    
    def add_documents(
        self,
        doc_ids: List[str],
        embeddings: np.ndarray,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        batch_size: int = 500
    ) -> None:
        """Add documents to the collection in batches."""
        n = len(doc_ids)
        print(f"Adding {n} documents to ChromaDB...")
        
        for i in range(0, n, batch_size):
            batch_end = min(i + batch_size, n)
            
            # ChromaDB metadata values must be str/int/float/bool
            # Serialize complex types (like cluster probability dicts) to JSON strings
            sanitized_meta = []
            for m in metadatas[i:batch_end]:
                clean = {}
                for k, v in m.items():
                    if isinstance(v, (str, int, float, bool)):
                        clean[k] = v
                    else:
                        clean[k] = json.dumps(v)
                sanitized_meta.append(clean)
            
            self.collection.add(
                ids=doc_ids[i:batch_end],
                embeddings=embeddings[i:batch_end].tolist(),
                documents=texts[i:batch_end],
                metadatas=sanitized_meta
            )
            print(f"  Added batch {i//batch_size + 1}: docs {i}-{batch_end}")
    
    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 10,
        where: Optional[Dict] = None
    ) -> Dict:

        kwargs = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances", "embeddings"]
        }
        if where:
            kwargs["where"] = where
        
        return self.collection.query(**kwargs)
    
    def count(self) -> int:
        return self.collection.count()
    
    def get_all_embeddings(self) -> Dict:
        """Retrieve all embeddings for clustering. Used once during ingest."""
        return self.collection.get(include=["embeddings", "metadatas", "documents"])
    
    def update_metadata(self, doc_id: str, metadata: Dict) -> None:
        """Update cluster assignment for a document after clustering."""
        sanitized = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                sanitized[k] = v
            else:
                sanitized[k] = json.dumps(v)
        self.collection.update(ids=[doc_id], metadatas=[sanitized])
    
    def reset(self) -> None:
        """Drop and recreate the collection."""
        self.client.delete_collection(CHROMA_COLLECTION)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )


# Module-level instance
vector_store = VectorStore()