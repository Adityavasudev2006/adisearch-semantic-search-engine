# Adisearch 🔍

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi)
![Flask](https://img.shields.io/badge/Flask-Web_UI-black?style=for-the-badge&logo=flask)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange?style=for-the-badge&logo=databricks)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-yellow?style=for-the-badge&logo=huggingface)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker)
![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> **Semantic search engine** with fuzzy clustering and intelligent semantic cache, built on the 20 Newsgroups corpus.

---

## 📖 Executive Summary

**Adisearch** is a production-grade semantic search system designed to retrieve topically relevant documents even when exact vocabulary doesn't match. It combines dense vector embeddings, persistent vector storage, and a cluster-indexed semantic cache to deliver fast, intelligent search over a large document corpus.

The system bridges the vocabulary gap between modern queries and a 1990s Usenet corpus — searching _"best GPU"_ correctly retrieves posts about _"video cards"_ — demonstrating how semantic embeddings outperform keyword-based retrieval.

---

## 🏗 System Architecture

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
     ├── SentenceTransformer Embeddings  (all-MiniLM-L6-v2, 384D)
     ├── ChromaDB Vector Search          (cosine similarity)
     ├── Fuzzy C-Means Clustering        (k=15, m=2.0)
     └── Cluster-Indexed Semantic Cache  (threshold=0.85)
```

### Component Responsibilities

| Component           | Role                                                                              |
| :------------------ | :-------------------------------------------------------------------------------- |
| **Flask Web UI**    | Browser interface for submitting queries, viewing results, and managing the cache |
| **FastAPI Backend** | Embedding, vector search, cache lookup, and ranked document retrieval             |
| **ChromaDB**        | Persistent cosine-space vector store for document embeddings                      |
| **Fuzzy C-Means**   | Soft document clustering, enabling multi-topic membership and O(N/K) cache lookup |
| **Semantic Cache**  | Paraphrase-aware cache that serves results for semantically equivalent queries    |

---

## 🛠 Tech Stack

| Component            | Technology              | Description                                       |
| :------------------- | :---------------------- | :------------------------------------------------ |
| **Embeddings**       | `all-MiniLM-L6-v2`      | 384D sentence embeddings, L2-normalized           |
| **Vector DB**        | ChromaDB                | Persistent cosine-space document store            |
| **Clustering**       | Fuzzy C-Means           | Soft cluster membership across 15 clusters        |
| **Cache**            | Custom semantic cache   | Cluster-indexed, cosine similarity threshold 0.85 |
| **Backend**          | FastAPI                 | REST API for search and cache management          |
| **Frontend**         | Flask + Jinja2          | Lightweight web UI for API interaction            |
| **Containerization** | Docker + Docker Compose | Multi-service orchestration                       |

---

## 📂 Project Structure

```
Adisearch/
├── data/                     ← 20 Newsgroups corpus (download separately)
│   ├── 20_newsgroups/
│   └── mini_newsgroups/
│
├── notebooks/                ← Exploration and clustering analysis
│
├── src/                      ← Core modules
│   ├── embedder.py           ← Sentence embedding wrapper
│   ├── vector_store.py       ← ChromaDB interface
│   ├── fuzzy_clustering.py   ← Fuzzy C-Means logic
│   ├── semantic_cache.py     ← Cluster-indexed cache
│   └── search_engine.py      ← Query orchestration
│
├── api/                      ← FastAPI backend
│   └── main.py
│
├── web/                      ← Flask frontend
│   ├── app.py
│   └── templates/
│       └── index.html
│
├── scripts/                  ← One-time data processing
│   ├── ingest.py             ← Dataset ingestion into ChromaDB
│   └── cluster.py            ← Fuzzy clustering pipeline
│
├── tests/                    ← Unit tests
│
├── chroma_db/                ← Persistent vector database
├── cache_data/               ← Persistent semantic cache
├── cluster_model/            ← Saved clustering pipeline
│
├── Dockerfile
├── docker-compose.yml
├── start.sh
├── requirements.txt
└── README.md
```

---

## 📥 Dataset

The corpus is based on the classic **20 Newsgroups** dataset, provided as part of this project.

📥 **[Download data.zip from Google Drive](https://drive.google.com/drive/folders/1TjK4mT0mAMZjubitk87GpzoTwVeJz7dc?usp=drive_link)**

After downloading:

1. Extract `data.zip`
2. Place the resulting `data/` folder in the project root

```
Adisearch/
└── data/          ← paste extracted folder here
    ├── 20_newsgroups/
    └── mini_newsgroups/
```

> **Note:** The `data/` folder is excluded from this repository. It must be downloaded separately before running any scripts.

---

## 🚀 Running the Application

You can run Adisearch in two ways:

### Option 1 — Local Development

#### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
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

#### 5. Start the backend API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

#### 6. Open the Web UI

```bash
http://localhost:5000
```

Open **[http://localhost:5000](http://localhost:5000)** in your browser.

---

### Option 2 — Docker

Build and launch both services with a single command:

```bash
docker compose up --build
```

| Service | URL                   |
| :------ | :-------------------- |
| Web UI  | http://localhost:5000 |
| API     | http://localhost:8000 |

Stop all services:

```bash
docker compose down
```

Persistent volumes ensure that `chroma_db/`, `cache_data/`, and `cluster_model/` are retained across container restarts.

---

## 🌐 Web Interface

A lightweight browser-based UI is included for interacting with the API without `curl`.

**Features:**

- Submit semantic search queries
- View ranked JSON results
- Inspect cache hit/miss statistics
- Flush the semantic cache

### Supported Actions

| Action      | Method   | Endpoint       |
| :---------- | :------- | :------------- |
| Run Query   | `POST`   | `/query`       |
| Cache Stats | `GET`    | `/cache/stats` |
| Flush Cache | `DELETE` | `/cache`       |

---

## 📡 API Reference

### `POST /query`

Runs semantic search over the document corpus.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Space shuttle launch nasa mission"}'
```

**Response:**

```json
{
  "query": "Space shuttle launch nasa mission",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": [...],
  "dominant_cluster": 3
}
```

---

### `GET /cache/stats`

Returns current cache performance statistics.

```bash
curl http://localhost:8000/cache/stats
```

**Response:**

```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
```

---

### `DELETE /cache`

Flushes all entries from the semantic cache.

```bash
curl -X DELETE http://localhost:8000/cache
```

---

### `GET /`

Health check endpoint.

```bash
curl http://localhost:8000/
```

---

## 🧠 Semantic Cache

Traditional caches rely on exact key matches. Adisearch implements a **semantic cache** that recognizes paraphrased queries by comparing embeddings.

**Example:**

```
Query 1: "Space shuttle launch nasa mission"
Query 2: "Space shuttle launch by nasa"
```

Although phrased differently, both queries produce similar embeddings and the second is served directly from cache — no redundant vector search required.

### Cache Design

- **Cosine similarity** comparison between query embeddings
- **Cluster-indexed lookup** — only entries within the same dominant cluster are compared
- **Similarity threshold:** `0.85`
- **Complexity:** O(N/K) vs O(N) for flat cache, where K = number of clusters

---

## 🔵 Fuzzy Clustering

Unlike hard clustering, **Fuzzy C-Means** assigns each document a _membership distribution_ across all clusters — allowing multi-topic documents to be represented accurately.

**Example** — a post from `talk.politics.guns` may receive:

```
Cluster 4 → 0.71  (politics)
Cluster 7 → 0.43  (firearms)
```

These memberships are used both for thematic analysis and to narrow the semantic cache search space, accelerating lookups for large caches.

---

## 📊 Notes on Similarity Scores

The 20 Newsgroups corpus consists of Usenet posts from the **early 1990s**. A natural vocabulary gap exists between modern query language and the historical documents.

| Modern Query Term | Corpus Equivalent |
| :---------------- | :---------------- |
| GPU               | video card        |
| PC                | workstation       |
| online            | network / usenet  |
| smartphone        | handheld device   |

Cosine similarity scores typically fall in the **0.45–0.60** range despite topical relevance being high. The sentence embedding model (`all-MiniLM-L6-v2`) successfully bridges this gap through semantic representation rather than surface-level token matching.

---

## 📓 Notebooks

See `notebooks/01_exploration_and_clustering.ipynb` for:

- Corpus exploration and statistics
- PCA dimensionality reduction
- Fuzzy C-Means cluster visualizations
- Cache performance analysis

---

## 📜 License

This project is licensed under the **MIT License**.  
© 2024 Aditya Vasudev K

---

<p align="center">Built with ❤️ using FastAPI · ChromaDB · HuggingFace · Docker</p>
