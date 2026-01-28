"""
Feature extraction for MLP router.
Compresses query embeddings and extracts routing features.
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Optional
from sklearn.decomposition import PCA


# Feature dimensions
PCA_COMPONENTS = 16
EMBEDDING_DIM = 1024

# Doc types for one-hot encoding
DOC_TYPES = ["policy", "contract", "legal", "technical", "general"]


class FeatureExtractor:
    """
    Extracts features for MLP router.
    
    Features (25 total):
    - Query embedding PCA (16)
    - Top chunk similarity (1)
    - Num relevant chunks (1)
    - Query token count normalized (1)
    - Doc token count normalized (1)
    - Doc type one-hot (5)
    """
    
    def __init__(self, pca_path: Optional[str] = None):
        self.pca: Optional[PCA] = None
        self.fitted = False
        
        if pca_path and Path(pca_path).exists():
            self.load_pca(pca_path)
    
    def fit_pca(self, embeddings: np.ndarray, n_components: int = PCA_COMPONENTS):
        """
        Fit PCA on training embeddings.
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
            n_components: Number of PCA components
        """
        self.pca = PCA(n_components=n_components)
        self.pca.fit(embeddings)
        self.fitted = True
        
        explained_var = sum(self.pca.explained_variance_ratio_) * 100
        print(f"✓ PCA fitted: {n_components} components explain {explained_var:.1f}% variance")
    
    def save_pca(self, path: str):
        """Save fitted PCA to file."""
        if not self.fitted:
            raise ValueError("PCA not fitted yet")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "wb") as f:
            pickle.dump(self.pca, f)
        
        print(f"✓ PCA saved to {path}")
    
    def load_pca(self, path: str):
        """Load PCA from file."""
        with open(path, "rb") as f:
            self.pca = pickle.load(f)
        self.fitted = True
        print(f"✓ PCA loaded from {path}")
    
    def transform_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Transform embedding using PCA.
        Falls back to truncation if PCA not fitted.
        """
        embedding = np.array(embedding).reshape(1, -1)
        
        if self.fitted and self.pca is not None:
            return self.pca.transform(embedding).flatten()
        else:
            # Fallback: use first 16 dimensions
            return embedding.flatten()[:PCA_COMPONENTS]
    
    def extract(
        self,
        query_embedding: list[float],
        top_similarities: list[float],
        query_token_count: int,
        doc_token_count: int,
        doc_type: str
    ) -> np.ndarray:
        """
        Extract full feature vector.
        
        Args:
            query_embedding: Raw embedding from mistral-embed (1024 dims)
            top_similarities: List of top-k chunk similarity scores
            query_token_count: Number of tokens in query
            doc_token_count: Total tokens in document
            doc_type: Document type string
        
        Returns:
            Feature array of shape (25,)
        """
        features = []
        
        # 1. PCA-compressed query embedding (16 dims)
        emb_pca = self.transform_embedding(np.array(query_embedding))
        features.extend(emb_pca.tolist())
        
        # 2. Top chunk similarity (1 dim)
        top_sim = max(top_similarities) if top_similarities else 0.0
        features.append(float(top_sim))
        
        # 3. Number of relevant chunks (sim > 0.5) (1 dim)
        num_relevant = sum(1 for s in top_similarities if s > 0.5)
        features.append(float(num_relevant))
        
        # 4. Query token count normalized (1 dim)
        features.append(query_token_count / 100.0)
        
        # 5. Doc token count normalized (1 dim)
        features.append(doc_token_count / 10000.0)
        
        # 6. Doc type one-hot (5 dims)
        doc_type_lower = doc_type.lower()
        onehot = [1.0 if doc_type_lower == dt else 0.0 for dt in DOC_TYPES]
        features.extend(onehot)
        
        return np.array(features, dtype=np.float32)
    
    def extract_batch(
        self,
        embeddings: list[list[float]],
        similarities: list[list[float]],
        query_tokens: list[int],
        doc_tokens: list[int],
        doc_types: list[str]
    ) -> np.ndarray:
        """
        Extract features for a batch of examples.
        
        Returns:
            Feature array of shape (n_samples, 25)
        """
        features = []
        
        for i in range(len(embeddings)):
            feat = self.extract(
                query_embedding=embeddings[i],
                top_similarities=similarities[i],
                query_token_count=query_tokens[i],
                doc_token_count=doc_tokens[i],
                doc_type=doc_types[i]
            )
            features.append(feat)
        
        return np.array(features, dtype=np.float32)


def extract_features_from_labeled_data(labeled_item: dict, extractor: FeatureExtractor) -> np.ndarray:
    """
    Extract features from a labeled data item.
    
    Args:
        labeled_item: Dict with query_embedding, top_similarities, etc.
        extractor: FeatureExtractor instance
    
    Returns:
        Feature array
    """
    return extractor.extract(
        query_embedding=labeled_item["query_embedding"],
        top_similarities=labeled_item["top_similarities"],
        query_token_count=labeled_item["query_token_count"],
        doc_token_count=labeled_item["doc_token_count"],
        doc_type=labeled_item["doc_type"]
    )


if __name__ == "__main__":
    # Test feature extraction
    print("Testing FeatureExtractor...")
    
    extractor = FeatureExtractor()
    
    # Create synthetic embeddings
    np.random.seed(42)
    train_embeddings = np.random.randn(100, EMBEDDING_DIM)
    
    # Fit PCA
    extractor.fit_pca(train_embeddings)
    
    # Test single extraction
    test_embedding = np.random.randn(EMBEDDING_DIM).tolist()
    features = extractor.extract(
        query_embedding=test_embedding,
        top_similarities=[0.85, 0.72, 0.61, 0.55, 0.43],
        query_token_count=15,
        doc_token_count=2500,
        doc_type="policy"
    )
    
    print(f"\nFeature vector shape: {features.shape}")
    print(f"Feature breakdown:")
    print(f"  PCA embedding (0-15): {features[:16][:3]}... (showing first 3)")
    print(f"  Top similarity (16): {features[16]:.3f}")
    print(f"  Num relevant (17): {features[17]:.0f}")
    print(f"  Query tokens norm (18): {features[18]:.3f}")
    print(f"  Doc tokens norm (19): {features[19]:.3f}")
    print(f"  Doc type one-hot (20-24): {features[20:25]}")
    
    # Test batch extraction
    batch_features = extractor.extract_batch(
        embeddings=[test_embedding, test_embedding],
        similarities=[[0.9, 0.8], [0.5, 0.4]],
        query_tokens=[10, 25],
        doc_tokens=[1000, 5000],
        doc_types=["policy", "contract"]
    )
    print(f"\nBatch features shape: {batch_features.shape}")
    
    # Test save/load
    extractor.save_pca("test_pca.pkl")
    extractor2 = FeatureExtractor("test_pca.pkl")
    
    features2 = extractor2.extract(
        query_embedding=test_embedding,
        top_similarities=[0.85, 0.72],
        query_token_count=15,
        doc_token_count=2500,
        doc_type="policy"
    )
    
    assert np.allclose(features[:16], features2[:16])
    print("✓ PCA save/load verification passed")
    
    # Cleanup
    Path("test_pca.pkl").unlink()
    print("\n✅ All tests passed!")
