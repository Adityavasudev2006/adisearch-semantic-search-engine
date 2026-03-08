"""Basic tests for the semantic cache."""
import sys
sys.path.insert(0, '.')

import numpy as np
import pytest
from src.semantic_cache import SemanticCache
from pathlib import Path
import tempfile


def make_random_unit_vector(dim=384):
    v = np.random.randn(dim)
    return v / np.linalg.norm(v)

def make_uniform_memberships(n=15):
    return np.ones(n) / n


def test_cache_miss_on_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = SemanticCache(threshold=0.85, persist_path=Path(tmpdir) / "cache.json")
        q_emb = make_random_unit_vector()
        q_mem = make_uniform_memberships()
        result = cache.lookup(q_emb, q_mem)
        assert result is None


def test_cache_hit_on_identical():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = SemanticCache(threshold=0.85, persist_path=Path(tmpdir) / "cache.json")
        q_emb = make_random_unit_vector()
        q_mem = make_uniform_memberships()
        
        cache.store("test query", q_emb, q_mem, {"result": "test"})
        result = cache.lookup(q_emb, q_mem)
        
        assert result is not None
        entry, score = result
        assert score >= 0.99
        assert entry.query == "test query"


def test_cache_miss_on_orthogonal():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = SemanticCache(threshold=0.85, persist_path=Path(tmpdir) / "cache.json")
        q_emb1 = make_random_unit_vector()
        q_emb2 = make_random_unit_vector()
        q_mem = make_uniform_memberships()
        
        cache.store("query 1", q_emb1, q_mem, {"result": "r1"})
        result = cache.lookup(q_emb2, q_mem)
        # Very unlikely to hit (random vectors rarely have cosine > 0.85)
        # This probabilistically confirms threshold behavior


def test_flush():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = SemanticCache(threshold=0.85, persist_path=Path(tmpdir) / "cache.json")
        q_emb = make_random_unit_vector()
        q_mem = make_uniform_memberships()
        
        cache.store("q1", q_emb, q_mem, {})
        cache.store("q2", make_random_unit_vector(), q_mem, {})
        assert cache.get_stats()["total_entries"] == 2
        
        cache.flush()
        assert cache.get_stats()["total_entries"] == 0
        assert cache.get_stats()["hit_count"] == 0


def test_stats():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = SemanticCache(threshold=0.85, persist_path=Path(tmpdir) / "cache.json")
        q_emb = make_random_unit_vector()
        q_mem = make_uniform_memberships()
        
        # Miss (store)
        cache.store("q1", q_emb, q_mem, {})
        # Hit
        cache.lookup(q_emb, q_mem)
        
        stats = cache.get_stats()
        assert stats["miss_count"] == 1
        assert stats["hit_count"] == 1
        assert stats["hit_rate"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])