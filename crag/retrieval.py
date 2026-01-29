"""
Hybrid Retrieval with Pinecone.

Uses Pinecone's native sparse-dense hybrid search for persistent BM25 + Semantic search.
No need to rebuild indices in memory - everything is stored in Pinecone.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pinecone import Pinecone, ServerlessSpec


@dataclass
class Document:
    """A document chunk for retrieval."""
    content: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0
    source: str = "vectorstore"
    id: Optional[str] = None
    
    def __hash__(self):
        return hash(self.content)
    
    def __eq__(self, other):
        if isinstance(other, Document):
            return self.content == other.content
        return False
    
    @property
    def page_content(self) -> str:
        """Compatibility with LangChain Document."""
        return self.content


class PineconeHybridRetriever:
    """
    Hybrid Retriever using Pinecone's sparse-dense search.
    
    Features:
    - Persistent storage: No need to rebuild indices
    - BM25-like sparse search via pinecone-text
    - Dense semantic search via Mistral embeddings
    - Native hybrid fusion in Pinecone
    
    Usage:
        retriever = PineconeHybridRetriever(
            index_name="crag-docs",
            namespace="my-project"
        )
        
        # Add documents (one-time indexing)
        retriever.add_documents(documents)
        
        # Query (uses persistent index)
        results = retriever.retrieve("What is GPT-4's MMLU score?", top_k=5)
    """
    
    def __init__(
        self,
        index_name: str = "crag-hybrid",
        namespace: str = "default",
        dense_model: str = "mistral-embed",
        sparse_alpha: float = 0.5,
        dimension: int = 1024,  # mistral-embed dimension
        create_index: bool = True,
    ):
        """
        Initialize Pinecone hybrid retriever.
        
        Args:
            index_name: Name of the Pinecone index
            namespace: Namespace within the index
            dense_model: Model for dense embeddings (default: mistral-embed)
            sparse_alpha: Weight for sparse vs dense (0=dense only, 1=sparse only)
            dimension: Embedding dimension (1024 for mistral-embed)
            create_index: Whether to create index if it doesn't exist
        """
        self.index_name = index_name
        self.namespace = namespace
        self.dense_model = dense_model
        self.sparse_alpha = sparse_alpha
        self.dimension = dimension
        
        # Lazy-init clients
        self._pinecone = None
        self._index = None
        self._mistral = None
        self._sparse_encoder = None
        
        if create_index:
            self._ensure_index()
    
    def _get_pinecone(self) -> Pinecone:
        """Get or create Pinecone client."""
        if self._pinecone is None:
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("PINECONE_API_KEY not found in environment")
            self._pinecone = Pinecone(api_key=api_key)
        return self._pinecone
    
    def _get_index(self):
        """Get or create Pinecone index."""
        if self._index is None:
            pc = self._get_pinecone()
            self._index = pc.Index(self.index_name)
        return self._index
    
    def _get_mistral(self):
        """Get or create Mistral client for embeddings."""
        if self._mistral is None:
            from mistralai import Mistral
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("MISTRAL_API_KEY not found in environment")
            self._mistral = Mistral(api_key=api_key)
        return self._mistral
    
    def _get_sparse_encoder(self):
        """Get or create sparse encoder (BM25)."""
        if self._sparse_encoder is None:
            from pinecone_text.sparse import BM25Encoder
            self._sparse_encoder = BM25Encoder.default()
        return self._sparse_encoder
    
    def _ensure_index(self):
        """Create index if it doesn't exist."""
        pc = self._get_pinecone()
        
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"Creating Pinecone index: {self.index_name}")
            pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="dotproduct",  # Required for hybrid search
                spec=ServerlessSpec(
                    cloud=os.getenv("PINECONE_CLOUD", "aws"),
                    region=os.getenv("PINECONE_REGION", "us-east-1")
                )
            )
            print(f"Index {self.index_name} created")
    
    def _get_dense_embedding(self, text: str) -> List[float]:
        """Get dense embedding from Mistral."""
        client = self._get_mistral()
        response = client.embeddings.create(
            model=self.dense_model,
            inputs=[text[:8000]]  # Truncate to avoid token limits
        )
        return response.data[0].embedding
    
    def _get_dense_embeddings_batch(self, texts: List[str], batch_size: int = 20) -> List[List[float]]:
        """Get dense embeddings in batches."""
        client = self._get_mistral()
        all_embeddings = []
        
        truncated = [t[:8000] for t in texts]
        
        for i in range(0, len(truncated), batch_size):
            batch = truncated[i:i + batch_size]
            response = client.embeddings.create(
                model=self.dense_model,
                inputs=batch
            )
            all_embeddings.extend([item.embedding for item in response.data])
        
        return all_embeddings
    
    def _get_sparse_embedding(self, text: str) -> Dict[str, Any]:
        """Get sparse (BM25) embedding."""
        encoder = self._get_sparse_encoder()
        sparse = encoder.encode_documents([text])[0]
        
        # Convert to Pinecone sparse format
        return {
            "indices": sparse["indices"],
            "values": sparse["values"]
        }
    
    def _get_sparse_query_embedding(self, text: str) -> Dict[str, Any]:
        """Get sparse embedding for query (uses query encoder)."""
        encoder = self._get_sparse_encoder()
        sparse = encoder.encode_queries([text])[0]
        
        return {
            "indices": sparse["indices"],
            "values": sparse["values"]
        }
    
    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 50,
        fit_sparse: bool = True
    ) -> int:
        """
        Add documents to Pinecone index.
        
        Args:
            documents: List of Document objects
            batch_size: Batch size for upserts
            fit_sparse: Whether to fit BM25 on documents (needed once)
            
        Returns:
            Number of documents added
        """
        if not documents:
            return 0
        
        index = self._get_index()
        
        # Fit sparse encoder on corpus (for BM25 IDF calculation)
        if fit_sparse:
            print("Fitting BM25 encoder on corpus...")
            encoder = self._get_sparse_encoder()
            corpus_texts = [doc.content for doc in documents]
            encoder.fit(corpus_texts)
            print("BM25 encoder fitted")
        
        # Get all dense embeddings
        print(f"Computing dense embeddings for {len(documents)} documents...")
        texts = [doc.content for doc in documents]
        dense_embeddings = self._get_dense_embeddings_batch(texts)
        
        # Prepare vectors for upsert
        vectors = []
        for i, doc in enumerate(documents):
            doc_id = doc.id or f"doc_{i}"
            
            # Get sparse embedding
            sparse = self._get_sparse_embedding(doc.content)
            
            vectors.append({
                "id": doc_id,
                "values": dense_embeddings[i],
                "sparse_values": sparse,
                "metadata": {
                    "content": doc.content[:40000],  # Pinecone metadata limit
                    **doc.metadata
                }
            })
        
        # Upsert in batches
        print(f"Upserting {len(vectors)} vectors to Pinecone...")
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch, namespace=self.namespace)
        
        print(f"Added {len(documents)} documents to index")
        return len(documents)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        alpha: Optional[float] = None,
        filter: Optional[dict] = None
    ) -> List[Document]:
        """
        Retrieve documents using hybrid search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Override sparse_alpha (0=dense only, 1=sparse only)
            filter: Optional metadata filter
            
        Returns:
            List of Document objects sorted by relevance
        """
        index = self._get_index()
        alpha = alpha if alpha is not None else self.sparse_alpha
        
        # Get query embeddings
        dense_query = self._get_dense_embedding(query)
        sparse_query = self._get_sparse_query_embedding(query)
        
        # Hybrid query
        results = index.query(
            namespace=self.namespace,
            vector=dense_query,
            sparse_vector=sparse_query,
            top_k=top_k,
            include_metadata=True,
            filter=filter
        )
        
        # Convert to Documents
        documents = []
        for match in results.matches:
            metadata = match.metadata or {}
            content = metadata.pop("content", "")
            
            doc = Document(
                content=content,
                metadata=metadata,
                score=match.score,
                source="vectorstore",
                id=match.id
            )
            documents.append(doc)
        
        return documents
    
    def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 5
    ) -> List[tuple]:
        """Retrieve documents with their scores."""
        docs = self.retrieve(query, top_k=top_k)
        return [(doc, doc.score) for doc in docs]
    
    def delete(self, ids: Optional[List[str]] = None, delete_all: bool = False):
        """Delete documents from index."""
        index = self._get_index()
        
        if delete_all:
            index.delete(delete_all=True, namespace=self.namespace)
        elif ids:
            index.delete(ids=ids, namespace=self.namespace)
    
    def get_stats(self) -> dict:
        """Get index statistics."""
        index = self._get_index()
        return index.describe_index_stats()


# ============== Wrapper for compatibility ==============

class HybridRetriever(PineconeHybridRetriever):
    """
    Alias for PineconeHybridRetriever for backward compatibility.
    
    Uses Pinecone's native hybrid search for persistent BM25 + Semantic.
    """
    pass


# ============== In-Memory Fallback (for testing without Pinecone) ==============

class InMemoryHybridRetriever:
    """
    In-memory hybrid retriever for testing/development.
    
    Uses local BM25 + semantic search with RRF fusion.
    NOT recommended for production - use PineconeHybridRetriever instead.
    """
    
    def __init__(self):
        self.documents: List[Document] = []
        self._embeddings = None
        self._bm25_fitted = False
        
    def _get_mistral(self):
        from mistralai import Mistral
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found")
        return Mistral(api_key=api_key)
    
    def add_documents(self, documents: List[Document]) -> int:
        self.documents = documents
        return len(documents)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Simple semantic-only retrieval for testing."""
        if not self.documents:
            return []
        
        client = self._get_mistral()
        
        # Get embeddings
        texts = [doc.content[:8000] for doc in self.documents]
        doc_response = client.embeddings.create(model="mistral-embed", inputs=texts)
        doc_embeddings = [item.embedding for item in doc_response.data]
        
        query_response = client.embeddings.create(model="mistral-embed", inputs=[query])
        query_embedding = query_response.data[0].embedding
        
        # Compute similarities
        import numpy as np
        doc_embeddings = np.array(doc_embeddings)
        query_embedding = np.array(query_embedding)
        
        similarities = np.dot(doc_embeddings, query_embedding) / (
            np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for i in top_indices:
            doc = self.documents[i]
            doc.score = float(similarities[i])
            results.append(doc)
        
        return results
