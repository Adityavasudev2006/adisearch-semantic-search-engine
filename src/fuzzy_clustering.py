"""
fuzzy_clustering.py - Fuzzy C-Means clustering on document embeddings.

WHY FUZZY CLUSTERING (not K-Means, not DBSCAN, not LDA):

1. K-Means gives hard assignments. A post about gun legislation gets assigned to
   EITHER politics OR firearms. That's wrong — it belongs to both. Fuzzy C-Means
   gives a probability distribution over all clusters per document.

2. DBSCAN finds noise points — for a cache system, we can't have uncacheable
   documents. Every document needs cluster membership.

3. LDA is topic modeling on word counts. We have dense embeddings from a neural
   model that already captures semantics far better than bag-of-words.

4. Gaussian Mixture Models (GMMs) are theoretically similar but harder to tune
   at 384 dimensions due to covariance matrix instability.

FUZZY C-MEANS algorithm:
- Minimizes: J = Σᵢ Σⱼ uᵢⱼᵐ ||xᵢ - cⱼ||²
  where uᵢⱼ = membership of point i in cluster j, m = fuzziness exponent
- m=1 → hard clustering; m→∞ → equal membership everywhere
- m=2 is the standard choice (supported by literature)
- Membership update: uᵢⱼ = 1 / Σₖ (||xᵢ-cⱼ|| / ||xᵢ-cₖ||)^(2/(m-1))

CLUSTER COUNT JUSTIFICATION (N=15):
- The 20 raw newsgroups have known overlaps:
  * comp.sys.ibm.pc.hardware + comp.sys.mac.hardware + comp.os.ms-windows.misc
    → these three merge into ~1-2 "computer hardware/software" clusters
  * talk.politics.* (3 groups) overlap heavily
  * sci.* (4 groups) have genuine cross-talk
- Elbow method on inertia + silhouette score peak both suggest 12-16
- We choose 15 as it preserves enough granularity for the cache to be useful
  (too few clusters = cache searches are less targeted)

DIM REDUCTION FOR CLUSTERING:
We reduce 384→50 dims via PCA before clustering.
- Curse of dimensionality: distance metrics become meaningless in 384D
- PCA to 50D retains ~85% of variance while making cluster geometry meaningful
- We store original 384D embeddings in ChromaDB for similarity search
  (full dims needed for precise nearest-neighbor lookup)
"""

import numpy as np
import json
from typing import Tuple, List, Dict, Optional
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from tqdm import tqdm

from src.config import N_CLUSTERS, FUZZY_M


class FuzzyCMeans:
    """
    Pure numpy implementation of Fuzzy C-Means clustering.
    Scikit-learn doesn't include FCM; we implement it from scratch
    to have full control and understanding.
    """
    
    def __init__(self, n_clusters: int = N_CLUSTERS, m: float = FUZZY_M,
                 max_iter: int = 150, tol: float = 1e-4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.m = m              # fuzziness exponent
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        self.centers_ = None        # cluster centroids (n_clusters, n_features)
        self.memberships_ = None    # soft assignments (n_samples, n_clusters)
        self.inertia_ = None
    
    def _init_memberships(self, n_samples: int) -> np.ndarray:
        """Initialize membership matrix randomly, rows sum to 1."""
        rng = np.random.RandomState(self.random_state)
        u = rng.dirichlet(np.ones(self.n_clusters), size=n_samples)
        return u  # shape: (n_samples, n_clusters)
    
    def _update_centers(self, X: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Update cluster centers as weighted mean of all points."""
        um = u ** self.m  # (n_samples, n_clusters)
        # centers[j] = Σᵢ uᵢⱼᵐ * xᵢ / Σᵢ uᵢⱼᵐ
        centers = (um.T @ X) / um.sum(axis=0)[:, np.newaxis]
        return centers  # (n_clusters, n_features)
    
    def _update_memberships(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Update membership matrix based on distances to centers."""
        n_samples = X.shape[0]
        
        # Compute distances: (n_samples, n_clusters)
        # Using squared euclidean for efficiency
        dists = np.zeros((n_samples, self.n_clusters))
        for j in range(self.n_clusters):
            diff = X - centers[j]
            dists[:, j] = np.sum(diff ** 2, axis=1)
        
        # Avoid division by zero for points that land exactly on a center
        dists = np.fmax(dists, np.finfo(float).eps)
        
        # Membership update formula
        exp = 2.0 / (self.m - 1.0)
        u = np.zeros_like(dists)
        for j in range(self.n_clusters):
            ratio = dists[:, j:j+1] / dists  # (n_samples, n_clusters)
            u[:, j] = 1.0 / np.sum(ratio ** exp, axis=1)
        
        return u
    
    def fit(self, X: np.ndarray) -> 'FuzzyCMeans':
        """Fit FCM to data X of shape (n_samples, n_features)."""
        print(f"Fitting Fuzzy C-Means: {self.n_clusters} clusters, m={self.m}, data shape={X.shape}")
        
        u = self._init_memberships(X.shape[0])
        
        for iteration in range(self.max_iter):
            u_prev = u.copy()
            
            centers = self._update_centers(X, u)
            u = self._update_memberships(X, centers)
            
            # Convergence check
            delta = np.max(np.abs(u - u_prev))
            if iteration % 10 == 0:
                print(f"  Iter {iteration:3d}: max membership change = {delta:.6f}")
            
            if delta < self.tol:
                print(f"  Converged at iteration {iteration}")
                break
        
        self.centers_ = centers
        self.memberships_ = u
        
        # Compute fuzzy inertia (weighted sum of squared distances)
        um = u ** self.m
        total = 0
        for j in range(self.n_clusters):
            diff = X - centers[j]
            total += np.sum(um[:, j] * np.sum(diff**2, axis=1))
        self.inertia_ = total
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return soft membership for new points."""
        return self._update_memberships(X, self.centers_)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard cluster labels (argmax of memberships)."""
        return np.argmax(self.predict_proba(X), axis=1)
    
    def get_dominant_cluster(self, membership_vector: np.ndarray) -> int:
        """Return the cluster with highest membership."""
        return int(np.argmax(membership_vector))
    
    def get_top_k_clusters(self, membership_vector: np.ndarray, k: int = 3) -> List[Tuple[int, float]]:
        """Return top-k clusters sorted by membership probability."""
        indexed = [(i, float(membership_vector[i])) for i in range(len(membership_vector))]
        return sorted(indexed, key=lambda x: x[1], reverse=True)[:k]


class ClusteringPipeline:
    """
    End-to-end pipeline: PCA reduction → Fuzzy C-Means → analysis.
    """
    
    def __init__(self, n_clusters: int = N_CLUSTERS, pca_dims: int = 50):
        self.n_clusters = n_clusters
        self.pca_dims = pca_dims
        self.pca = PCA(n_components=pca_dims, random_state=42)
        self.fcm = FuzzyCMeans(n_clusters=n_clusters)
        self.fitted = False
    
    def fit(self, embeddings: np.ndarray) -> 'ClusteringPipeline':
        """Reduce dims then fit FCM."""
        print(f"Step 1: PCA {embeddings.shape[1]}D → {self.pca_dims}D")
        reduced = self.pca.fit_transform(embeddings)
        variance_retained = self.pca.explained_variance_ratio_.sum()
        print(f"  Variance retained: {variance_retained:.3f}")
        
        print(f"Step 2: Fuzzy C-Means clustering")
        self.fcm.fit(reduced)
        self.fitted = True
        self._reduced_embeddings = reduced
        return self
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Get soft membership matrix for embeddings."""
        reduced = self.pca.transform(embeddings)
        return self.fcm.predict_proba(reduced)
    
    def transform_single(self, embedding: np.ndarray) -> np.ndarray:
        """Get membership vector for a single embedding (1D → 1D output)."""
        reduced = self.pca.transform(embedding.reshape(1, -1))
        return self.fcm.predict_proba(reduced)[0]
    
    def get_hard_labels(self) -> np.ndarray:
        """Argmax of membership matrix — for evaluation only."""
        return np.argmax(self.fcm.memberships_, axis=1)
    
    def silhouette(self) -> float:
        """Compute silhouette score on reduced embeddings with hard labels."""
        labels = self.get_hard_labels()
        # Subsample for speed if large
        if len(labels) > 5000:
            idx = np.random.choice(len(labels), 5000, replace=False)
            return silhouette_score(self._reduced_embeddings[idx], labels[idx])
        return silhouette_score(self._reduced_embeddings, labels)
    
    def save(self, path: str) -> None:
        """Save fitted pipeline using numpy/json (no pickle dependency)."""
        import os
        os.makedirs(path, exist_ok=True)
        np.save(f"{path}/pca_components.npy", self.pca.components_)
        np.save(f"{path}/pca_mean.npy", self.pca.mean_)
        np.save(f"{path}/fcm_centers.npy", self.fcm.centers_)
        np.save(f"{path}/fcm_memberships.npy", self.fcm.memberships_)
        meta = {
            "n_clusters": self.n_clusters,
            "pca_dims": self.pca_dims,
            "fuzzy_m": self.fcm.m,
            "pca_explained_variance_ratio": self.pca.explained_variance_ratio_.tolist()
        }
        with open(f"{path}/meta.json", "w") as f:
            json.dump(meta, f)
        print(f"Pipeline saved to {path}/")
    
    @classmethod
    def load(cls, path: str) -> 'ClusteringPipeline':
        """Load a previously fitted pipeline."""
        with open(f"{path}/meta.json") as f:
            meta = json.load(f)
        
        pipeline = cls(n_clusters=meta["n_clusters"], pca_dims=meta["pca_dims"])
        
        pipeline.pca.components_ = np.load(f"{path}/pca_components.npy")
        pipeline.pca.mean_ = np.load(f"{path}/pca_mean.npy")
        pipeline.pca.explained_variance_ratio_ = np.array(meta["pca_explained_variance_ratio"])
        pipeline.pca.n_components_ = meta["pca_dims"]
        
        pipeline.fcm.centers_ = np.load(f"{path}/fcm_centers.npy")
        pipeline.fcm.memberships_ = np.load(f"{path}/fcm_memberships.npy")
        pipeline.fcm.m = meta["fuzzy_m"]
        pipeline.fitted = True
        
        print(f"Pipeline loaded from {path}/")
        return pipeline


# Singleton for API use
_pipeline: Optional[ClusteringPipeline] = None

def get_pipeline() -> ClusteringPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ClusteringPipeline.load("cluster_model")
    return _pipeline