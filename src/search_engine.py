# search_engine.py - Ties embedding + vector store + cache together.

import numpy as np
from typing import Dict, List, Any, Optional
import re

from src.embedder import embedder
from src.vector_store import vector_store
from src.semantic_cache import semantic_cache
from src.fuzzy_clustering import get_pipeline


def execute_query(query: str, n_results: int = 5) -> Dict[str, Any]:
    """
    Full query pipeline:
    1. Embed the query
    2. Get cluster memberships
    3. Check semantic cache
    4. If hit: return cached result
    5. If miss: search vector store, cache result, return
    
    Returns a dict matching the API response schema.
    """
    # Step 1: Embed query
    query_embedding = embedder.embed_single(query)
    
    # Step 2: Get cluster memberships from fitted clustering pipeline
    try:
        pipeline = get_pipeline()
        membership_vector = pipeline.transform_single(query_embedding)
        dominant_cluster = int(np.argmax(membership_vector))
    except Exception as e:
        # Fallback if clustering model not yet trained
        membership_vector = np.ones(15) / 15  # uniform
        dominant_cluster = 0
    
    # Step 3: Check semantic cache
    cache_result = semantic_cache.lookup(query_embedding, membership_vector)
    
    if cache_result is not None:
        entry, similarity_score = cache_result
        return {
            "query": query,
            "cache_hit": True,
            "matched_query": entry.query,
            "similarity_score": round(similarity_score, 4),
            "result": entry.result,
            "dominant_cluster": entry.dominant_cluster
        }
    
    # Step 4: Cache miss — execute real search
    results = _search_corpus(query_embedding, dominant_cluster, n_results)
    
    # Step 5: Store in cache
    semantic_cache.store(
        query=query,
        query_embedding=query_embedding,
        membership_vector=membership_vector,
        result=results
    )
    
    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": results,
        "dominant_cluster": dominant_cluster
    }


def _format_snippet(text: str, max_chars: int = 400) -> str:
    """
    Format raw newsgroup text into a clean, readable snippet.
    This does NOT modify stored data — only the displayed output.
    
    What we clean:
    - Literal \n sequences → proper newlines
    - Lines starting with > (quoted replies)
    - Leftover header-like lines (From:, Path:, etc.)
    - Multiple blank lines → single break
    - Leading/trailing whitespace
    """
    # Decode literal \n if stored as escaped string
    text = text.replace('\\n', '\n')
    
    lines = text.split('\n')
    clean_lines = []
    
    skip_prefixes = (
    'from:', 'path:', 'newsgroups:', 'subject:', 'message-id:',
    'date:', 'organization:', 'lines:', 'references:', 'sender:',
    'archive-name:', 'last-modified:', 'reply-to:'
    )
    
    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        # Skip quoted replies
        if stripped.startswith('>') or stripped.startswith('|'):
            continue

        # Skip header metadata
        if stripped.lower().startswith(skip_prefixes):
            continue

        # Skip email / address lines
        if '@' in stripped:
            continue

        # Skip attribution lines
        if re.match(r'^In article.*wrote:', stripped, re.IGNORECASE):
            continue

        # Skip all-uppercase section headings
        if stripped.isupper() and len(stripped.split()) > 3:
            continue

        # Skip common metadata keywords
        if 'archive-name' in stripped.lower() or 'last-modified' in stripped.lower():
            continue

        # Skip signature separators
        if stripped in ('--', '---'):
            continue

        clean_lines.append(stripped)
    
    # After building clean_lines, separate first line as title if it looks like one
    if clean_lines and len(clean_lines[0].split()) <= 10:
        title = clean_lines[0].rstrip('.')
        body_lines = clean_lines[1:]
    else:
        title = None
        body_lines = clean_lines

    body = ' '.join(body_lines)
    body = re.sub(r' {2,}', ' ', body).strip()

    # Only show title if body is substantial enough
    if title and len(body) > 80:
        result = f"{title} — {body}"
    else:
        result = body or (title or '')

    # Minimum snippet guard — if too short, it's noise
    if len(result.split()) < 10:
        result = ' '.join(clean_lines)
        result = re.sub(r' {2,}', ' ', result).strip()

    # Final truncation
    if len(result) > max_chars:
        result = result[:max_chars].rsplit(' ', 1)[0] + '...'

    return result


def _search_corpus(
    query_embedding: np.ndarray,
    dominant_cluster: int,
    n_results: int = 5
) -> List[Dict]:
    """
    Search ChromaDB for relevant documents.
    Returns list of result dicts.
    """
    try:
        chroma_results = vector_store.query(
            query_embedding=query_embedding,
            n_results=n_results
        )
        
        results = []
        if chroma_results and chroma_results.get("documents"):
            docs = chroma_results["documents"][0]
            metas = chroma_results["metadatas"][0]
            distances = chroma_results["distances"][0]
            ids = chroma_results["ids"][0]
            
            for doc, meta, dist, doc_id in zip(docs, metas, distances, ids):
                results.append({
                    "doc_id": doc_id,
                    "category": meta.get("category", "unknown"),
                    "text_snippet": _format_snippet(doc) if doc else "",
                    "similarity": round(1 - dist, 4),  # ChromaDB cosine distance → similarity
                    "cluster": meta.get("dominant_cluster", -1)
                })
        
        return results
    
    except Exception as e:
        return [{"error": str(e), "doc_id": "none", "category": "none", "text_snippet": "", "similarity": 0.0, "cluster": -1}]



