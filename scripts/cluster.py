"""
scripts/cluster.py - Run fuzzy clustering on the embedded corpus.
Run this AFTER ingest.py.

Usage: python scripts/cluster.py
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import json
from pathlib import Path
from src.fuzzy_clustering import ClusteringPipeline
from src.vector_store import vector_store
from src.config import N_CLUSTERS


def run_clustering():
    print("=" * 60)
    print("ADISEARCH — Fuzzy Clustering")
    print("=" * 60)
    
    # Load pre-computed embeddings
    if not Path("embeddings_cache.npy").exists():
        print("ERROR: embeddings_cache.npy not found. Run ingest.py first.")
        sys.exit(1)
    
    print("\n[1/4] Loading embeddings...")
    embeddings = np.load("embeddings_cache.npy")
    doc_ids = np.load("doc_ids_cache.npy", allow_pickle=True)
    print(f"Loaded: {embeddings.shape}")
    
    # Explore cluster counts (elbow method)
    print("\n[2/4] Finding optimal cluster count (elbow method)...")
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA
    
    # Reduce for speed during exploration
    pca_explore = PCA(n_components=50, random_state=42)
    reduced = pca_explore.fit_transform(embeddings)
    
    scores = {}
    for k in [8, 10, 12, 15, 18, 20]:
        pipeline_test = ClusteringPipeline(n_clusters=k)
        pipeline_test.fit(embeddings)
        sil = pipeline_test.silhouette()
        scores[k] = sil
        print(f"  k={k:2d}: silhouette={sil:.4f}, inertia={pipeline_test.fcm.inertia_:.2f}")
    
    best_k = max(scores, key=scores.get)
    print(f"\n  Best k by silhouette: {best_k}")
    print(f"  Using configured k={N_CLUSTERS} (see config.py for justification)")
    
    # Fit final pipeline
    print(f"\n[3/4] Fitting final pipeline (k={N_CLUSTERS})...")
    pipeline = ClusteringPipeline(n_clusters=N_CLUSTERS)
    pipeline.fit(embeddings)
    
    print(f"  Final silhouette score: {pipeline.silhouette():.4f}")
    print(f"  Cluster sizes (hard assignment):")
    labels = pipeline.get_hard_labels()
    from collections import Counter
    for cluster_id, count in sorted(Counter(labels).items()):
        print(f"    Cluster {cluster_id:2d}: {count} documents")
    
    # Save pipeline
    print("\n[4/4] Saving pipeline and updating ChromaDB metadata...")
    pipeline.save("cluster_model")
    
    # Update ChromaDB with cluster assignments
    memberships = pipeline.fcm.memberships_
    print("  Updating ChromaDB metadata (this may take a while)...")
    
    batch_size = 200
    for i in range(0, len(doc_ids), batch_size):
        batch_ids = doc_ids[i:i+batch_size]
        batch_memberships = memberships[i:i+batch_size]
        
        for doc_id, membership in zip(batch_ids, batch_memberships):
            dominant = int(np.argmax(membership))
            top3 = sorted(enumerate(membership.tolist()), key=lambda x: x[1], reverse=True)[:3]
            
            vector_store.update_metadata(
                doc_id=str(doc_id),
                metadata={
                    "dominant_cluster": dominant,
                    "cluster_probs": json.dumps({str(c): round(p, 4) for c, p in top3})
                }
            )
        
        if (i // batch_size) % 10 == 0:
            print(f"  Updated {min(i+batch_size, len(doc_ids))}/{len(doc_ids)}")
    
    print("\n✅ Clustering complete!")
    print(f"   Model saved to: cluster_model/")
    print(f"   Run the notebook for semantic analysis and visualizations")


if __name__ == "__main__":
    run_clustering()