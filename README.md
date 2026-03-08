# Adisearch 🔍

> Semantic search engine with fuzzy clustering and intelligent cache on 20 Newsgroups

## Dataset

The corpus used in this project was provided by **Trademarkia** as part of an AI/ML Engineer assignment.

It is based on the classic **20 Newsgroups** dataset and is required to run the ingestion and clustering pipeline.

📥 **[Download data.zip from Google Drive](https://drive.google.com/drive/folders/1TjK4mT0mAMZjubitk87GpzoTwVeJz7dc?usp=drive_link)**

After downloading:

1. Extract `data.zip`
2. Place the resulting `data/` folder in the root of the project directory

```
Adisearch/
├── data/          ← paste extracted folder here
│   ├── 20_newsgroups/
│   └── mini_newsgroups/
```

> **Note:** The `data/` folder is not included in this repository. You must download it separately using the link above before running any scripts.

## Project Structure

```
Adisearch/
├── data/                     ← 20 Newsgroups corpus
├── notebooks/                ← Exploration and clustering analysis
├── src/                      ← Core modules
│   ├── embedder.py
│   ├── vector_store.py
│   ├── fuzzy_clustering.py
│   ├── semantic_cache.py
│   └── search_engine.py
│
├── api/                      ← FastAPI backend (semantic search API)
│   └── main.py
│
├── web/                      ← Flask frontend (web interface)
│   ├── app.py
│   └── templates/
│       └── index.html
│
├── scripts/                  ← One-time data processing
│   ├── ingest.py
│   └── cluster.py
│
├── tests/                    ← Unit tests
│
├── chroma_db/                ← Persistent vector database
├── cache_data/               ← Persistent semantic cache
├── cluster_model/            ← Saved clustering pipeline
│
├── Dockerfile                ← Container build instructions
├── docker-compose.yml        ← Multi-service container orchestration
├── start.sh                  ← Startup script for running backend + frontend
│
├── requirements.txt
└── README.md
```

## System Architecture

The system consists of three layers:

```
User Browser
     │
     ▼
Flask Web UI (port 5000)
     │
     ▼
FastAPI Backend (port 8000)
     │
     ▼
Semantic Search Pipeline
     │
     ├── SentenceTransformer embeddings
     ├── ChromaDB vector search
     ├── Fuzzy C-Means clustering
     └── Cluster-indexed semantic cache
```

### Component Responsibilities

**Flask UI**

- Provides a web interface for interacting with the API
- Allows users to run POST, GET, and DELETE requests
- Displays raw JSON responses

**FastAPI Backend**

- Embeds queries
- Performs semantic search
- Handles cache lookups
- Returns ranked documents

## Running the Application

You can run Adisearch in two ways:

1️⃣ Local development
2️⃣ Docker container

---

### Option 1 — Run locally

#### 1. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

#### 2. Install dependencies

```bash
pip install -r requirements.txt
```

#### 3. Ingest the dataset

```bash
python scripts/ingest.py
```

#### 4. Run clustering

```bash
python scripts/cluster.py
```

#### 5. Start backend API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

#### 6. Start the web interface

```bash
python web/app.py
```

#### 7. Open the UI

```
http://localhost:5000
```

---

### Option 2 — Run with Docker

Build and start the containers:

```bash
docker compose up --build
```

This automatically launches:

- FastAPI backend → **port 8000**
- Flask web UI → **port 5000**

Open the web interface:

```
http://localhost:5000
```

To stop the containers:

```bash
docker compose down
```

## Web Interface

A lightweight web interface is included for interacting with the API.

**Features:**

- Submit semantic search queries
- View cache statistics
- Flush the semantic cache
- Display formatted JSON responses

### Supported Actions

| Action      | Endpoint         |
| ----------- | ---------------- |
| Run Query   | POST /query      |
| Cache Stats | GET /cache/stats |
| Flush Cache | DELETE /cache    |

Example JSON query:

```json
{
  "query": "Space shuttle launch nasa mission"
}
```

## API Endpoints

### POST /query

Runs semantic search.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Best graphics card for video games"}'
```

Response:

```json
{
  "query": "...",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": [...],
  "dominant_cluster": 3
}
```

---

### GET /cache/stats

Returns cache statistics.

```bash
curl http://localhost:8000/cache/stats
```

Response:

```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
```

---

### DELETE /cache

Flushes the semantic cache.

```bash
curl -X DELETE http://localhost:8000/cache
```

---

### GET / (health check)

```bash
curl http://localhost:8000/
```

## Architecture Overview

- **Embeddings**: `all-MiniLM-L6-v2` (384D, L2-normalized)
- **Vector DB**: ChromaDB (persistent, cosine space)
- **Clustering**: Fuzzy C-Means (k=15, m=2.0) on PCA-reduced embeddings
- **Cache**: Custom cluster-indexed semantic cache (threshold=0.85)

See `notebooks/01_exploration_and_clustering.ipynb` for full analysis and visualizations.

## Semantic Cache

Traditional caches rely on exact key matches. This project implements a **semantic cache** capable of recognizing paraphrased queries.

Example:

```
Query 1: "Best graphics cards"
Query 2: "Which video cards are good?"
```

Although phrased differently, the queries share similar embeddings and are served from cache.

### Cache Design

- Cosine similarity comparison
- Cluster-indexed lookup
- Similarity threshold: **0.85**

Cluster indexing drastically reduces lookup cost for large caches — O(N/K) instead of O(N), where K is the number of clusters.

## Fuzzy Clustering

Instead of assigning each document to a single cluster, this project uses **Fuzzy C-Means**, which allows documents to belong to multiple topics simultaneously.

Example — a post from `talk.politics.guns` may receive membership in:

```
Cluster 4 → politics
Cluster 7 → firearms
```

Each document receives a **membership distribution** across all clusters. These memberships are also used to accelerate semantic cache lookup by narrowing the candidate search space.

## Notes on Similarity Scores

The 20 Newsgroups dataset originates from Usenet posts written in the early 1990s. Because the corpus predates modern terminology, there is a natural vocabulary gap between contemporary queries and historical documents.

| Modern Query Term | Corpus Terminology |
| ----------------- | ------------------ |
| GPU               | video card         |
| PC                | workstation        |
| online            | network / usenet   |

As a result, cosine similarity scores may appear lower (typically ~0.45–0.60) despite the retrieved documents being topically relevant. The semantic embedding model successfully bridges this vocabulary gap, retrieving documents about the correct topic even when exact terminology differs.

## Docker Deployment

The project includes Docker support for running the full system in a containerized environment.

The Docker setup runs both:

- **FastAPI backend** (port 8000)
- **Flask web interface** (port 5000)

### Start the system

```bash
docker compose up --build
```

### Stop the system

```bash
docker compose down
```

### Services exposed

| Service | URL                   |
| ------- | --------------------- |
| Web UI  | http://localhost:5000 |
| API     | http://localhost:8000 |

Persistent volumes ensure that the following data is retained across container restarts:

- `chroma_db/` → vector database
- `cache_data/` → semantic cache
- `cluster_model/` → clustering model
