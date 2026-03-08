# semantic_cache.py - Handbuilt semantic cache. NO Redis, NO caching libraries.

import json
import time
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field, asdict

from src.config import CACHE_SIMILARITY_THRESHOLD, CACHE_DIR, N_CLUSTERS


@dataclass
class CacheEntry:
    """A single entry in the semantic cache."""
    query: str                          # original query text
    query_embedding: List[float]        # 384D embedding as list (JSON serializable)
    cluster_memberships: List[float]    # 15D soft membership vector
    dominant_cluster: int               # argmax of cluster_memberships
    result: Any                         # the cached result (search results list)
    timestamp: float = field(default_factory=time.time)
    hit_count: int = 0                  # how many times this entry was served from cache


class SemanticCache:
    
    def __init__(
        self,
        threshold: float = CACHE_SIMILARITY_THRESHOLD,
        persist_path: Optional[Path] = None,
        n_clusters: int = N_CLUSTERS
    ):
        self.threshold = threshold
        self.n_clusters = n_clusters
        self.persist_path = persist_path or (CACHE_DIR / "cache.json")
        
        # Core data structures
        self._entries: List[CacheEntry] = []
        
        # Cluster index: maps cluster_id → list of entry indices
        # This is the efficiency structure that makes large-cache lookup fast
        self._cluster_index: Dict[int, List[int]] = {i: [] for i in range(n_clusters)}
        
        # Stats
        self._hit_count: int = 0
        self._miss_count: int = 0
        
        # Load from disk if available
        self._load()
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))
    
    def _get_candidate_indices(self, membership_vector: np.ndarray, secondary_threshold: float = 0.05) -> List[int]:
        candidate_indices = set()
        
        for cluster_id, membership in enumerate(membership_vector):
            if membership >= secondary_threshold:
                candidate_indices.update(self._cluster_index.get(cluster_id, []))
        
        # Always include dominant cluster even if below threshold
        dominant = int(np.argmax(membership_vector))
        candidate_indices.update(self._cluster_index.get(dominant, []))
        
        return list(candidate_indices)
    
    def lookup(
        self,
        query_embedding: np.ndarray,
        membership_vector: np.ndarray
    ) -> Optional[Tuple[CacheEntry, float]]:
        candidate_indices = self._get_candidate_indices(membership_vector)
        
        if not candidate_indices:
            return None
        
        best_score = -1.0
        best_entry = None
        q_vec = np.array(query_embedding)
        
        for idx in candidate_indices:
            entry = self._entries[idx]
            entry_vec = np.array(entry.query_embedding)
            score = self._cosine_similarity(q_vec, entry_vec)
            
            if score > best_score:
                best_score = score
                best_entry = (entry, idx)
        
        if best_score >= self.threshold:
            # Cache hit!
            entry, idx = best_entry
            entry.hit_count += 1
            self._hit_count += 1
            return entry, best_score
        
        return None
    
    def store(
        self,
        query: str,
        query_embedding: np.ndarray,
        membership_vector: np.ndarray,
        result: Any
    ) -> CacheEntry:
        dominant_cluster = int(np.argmax(membership_vector))
        
        entry = CacheEntry(
            query=query,
            query_embedding=query_embedding.tolist(),
            cluster_memberships=membership_vector.tolist(),
            dominant_cluster=dominant_cluster,
            result=result
        )
        
        entry_idx = len(self._entries)
        self._entries.append(entry)
        
        # Update cluster index for ALL clusters with significant membership
        for cluster_id, membership in enumerate(membership_vector):
            if membership >= 0.05:  # index if 5%+ membership in this cluster
                self._cluster_index[cluster_id].append(entry_idx)
        
        self._miss_count += 1
        self._persist()
        
        return entry
    
    def flush(self) -> None:
        """Clear all cache entries and reset stats."""
        self._entries = []
        self._cluster_index = {i: [] for i in range(self.n_clusters)}
        self._hit_count = 0
        self._miss_count = 0
        self._persist()
    
    def get_stats(self) -> Dict:
        total = self._hit_count + self._miss_count
        return {
            "total_entries": len(self._entries),
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": round(self._hit_count / total, 4) if total > 0 else 0.0,
            "threshold": self.threshold,
            "cluster_distribution": {
                str(k): len(v) for k, v in self._cluster_index.items()
            }
        }
    
    def _persist(self) -> None:
        """Save cache to disk as JSON."""
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "threshold": self.threshold,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "entries": [asdict(e) for e in self._entries],
            "cluster_index": {str(k): v for k, v in self._cluster_index.items()}
        }
        
        with open(self.persist_path, 'w') as f:
            json.dump(data, f)
    
    def _load(self) -> None:
        """Load cache from disk if it exists."""
        if not self.persist_path.exists():
            return
        
        try:
            with open(self.persist_path) as f:
                data = json.load(f)
            
            self._hit_count = data.get("hit_count", 0)
            self._miss_count = data.get("miss_count", 0)
            
            self._entries = [
                CacheEntry(**e) for e in data.get("entries", [])
            ]
            
            self._cluster_index = {
                int(k): v for k, v in data.get("cluster_index", {}).items()
            }
            # Ensure all cluster keys exist
            for i in range(self.n_clusters):
                if i not in self._cluster_index:
                    self._cluster_index[i] = []
            
            print(f"Cache loaded: {len(self._entries)} entries, "
                  f"{self._hit_count} hits, {self._miss_count} misses")
        
        except Exception as e:
            print(f"Warning: Could not load cache ({e}). Starting fresh.")


# Module-level cache singleton
semantic_cache = SemanticCache()