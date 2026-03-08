from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any
import uvicorn
import threading
from web.app import app as flask_app
from src.search_engine import execute_query
from src.semantic_cache import semantic_cache
from src.fuzzy_clustering import get_pipeline
from src.embedder import embedder


# ─── Startup / Shutdown ─────────────────────────────────
def start_flask():
    flask_app.run(port=5000, debug=False, use_reloader=False)

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("🚀 Adisearch API starting up...")

    print("🚀 Adisearch API starting up...")

    # Start Flask UI
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()
    print("🌐 Flask UI running on http://localhost:5000")

    _ = embedder
    print("✅ Embedding model loaded")

    try:
        pipeline = get_pipeline()
        print(f"✅ Clustering pipeline loaded ({pipeline.n_clusters} clusters)")
    except Exception as e:
        print(f"⚠️ Clustering pipeline not found: {e}")

    print(f"✅ Semantic cache loaded ({semantic_cache.get_stats()['total_entries']} entries)")

    yield

    print("👋 Adisearch shutting down...")


# ─── FastAPI App ───────────────────────────────────────

app = FastAPI(
    title="Adisearch",
    description="Semantic search engine with fuzzy clustering and semantic cache",
    version="1.0.0",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ─── Request Models ───────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    n_results: int = 5


class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str]
    similarity_score: Optional[float]
    result: Any
    dominant_cluster: int


class CacheStatsResponse(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float


class DeleteResponse(BaseModel):
    message: str
    entries_cleared: int


# ─── Endpoints ───────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    result = execute_query(request.query, n_results=request.n_results)

    return QueryResponse(**result)


@app.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats():

    stats = semantic_cache.get_stats()

    return CacheStatsResponse(**stats)


@app.delete("/cache", response_model=DeleteResponse)
async def flush_cache():

    entries_before = semantic_cache.get_stats()["total_entries"]

    semantic_cache.flush()

    return DeleteResponse(
        message="Cache flushed successfully",
        entries_cleared=entries_before
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


# ─── Entry Point ───────────────────────────────────────

if __name__ == "__main__":

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )