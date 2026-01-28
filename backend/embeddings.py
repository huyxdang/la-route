"""
Embeddings module for Le-Route.
Handles text chunking, embedding generation, and vector storage.
"""

import os
import uuid
import numpy as np
from datetime import datetime
from typing import Optional
from mistralai import Mistral

from .models import Chunk, SessionData, DocType


# Mistral embedding model
EMBED_MODEL = "mistral-embed"
EMBED_DIMENSION = 1024

# Chunking parameters
DEFAULT_CHUNK_SIZE = 500  # tokens (approximate)
DEFAULT_CHUNK_OVERLAP = 100  # tokens (approximate)
CHARS_PER_TOKEN = 4  # Rough approximation


class TextChunker:
    """Splits text into overlapping chunks."""
    
    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Convert tokens to characters (approximation)
        self.chunk_chars = chunk_size * CHARS_PER_TOKEN
        self.overlap_chars = chunk_overlap * CHARS_PER_TOKEN
    
    def chunk_text(self, text: str) -> list[Chunk]:
        """
        Split text into overlapping chunks.
        Returns list of Chunk objects with character positions.
        """
        chunks = []
        text = text.strip()
        
        if not text:
            return chunks
        
        # If text is smaller than chunk size, return as single chunk
        if len(text) <= self.chunk_chars:
            return [Chunk(
                chunk_id=0,
                text=text,
                tokens=len(text) // CHARS_PER_TOKEN,
                start_char=0,
                end_char=len(text)
            )]
        
        start = 0
        chunk_id = 0
        
        while start < len(text):
            # Find end position
            end = min(start + self.chunk_chars, len(text))
            
            # Try to break at sentence boundary if not at end
            if end < len(text):
                # Look for sentence endings in last 20% of chunk
                search_start = start + int(self.chunk_chars * 0.8)
                search_region = text[search_start:end]
                
                # Find last sentence boundary
                for sep in ['. ', '.\n', '? ', '!\n', '! ', '?\n']:
                    last_sep = search_region.rfind(sep)
                    if last_sep != -1:
                        end = search_start + last_sep + len(sep)
                        break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    tokens=len(chunk_text) // CHARS_PER_TOKEN,
                    start_char=start,
                    end_char=end
                ))
                chunk_id += 1
            
            # Move start position with overlap
            start = end - self.overlap_chars
            if start >= len(text) - self.overlap_chars:
                break
        
        return chunks


class EmbeddingService:
    """Handles embedding generation using Mistral API."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment")
        self.client = Mistral(api_key=self.api_key)
    
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.
        Returns list of embedding vectors.
        """
        if not texts:
            return []
        
        # Mistral embed supports batching
        response = self.client.embeddings.create(
            model=EMBED_MODEL,
            inputs=texts
        )
        
        return [item.embedding for item in response.data]
    
    def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        embeddings = self.embed_texts([text])
        return embeddings[0] if embeddings else []


class VectorStore:
    """
    In-memory vector store using NumPy.
    Simple cosine similarity search.
    """
    
    def __init__(self):
        self.sessions: dict[str, SessionData] = {}
        self.embeddings: dict[str, np.ndarray] = {}  # session_id -> (n_chunks, embed_dim)
    
    def add_session(
        self,
        session_id: str,
        doc_type: DocType,
        original_text: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        title: Optional[str] = None
    ) -> SessionData:
        """Store a new document session with embeddings."""
        
        # Convert embeddings to numpy array
        embed_array = np.array(embeddings, dtype=np.float32)
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(embed_array, axis=1, keepdims=True)
        embed_array = embed_array / (norms + 1e-10)
        
        # Update chunks with embeddings
        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i]
        
        # Create session data
        session = SessionData(
            session_id=session_id,
            doc_type=doc_type,
            title=title,
            original_text=original_text,
            chunks=chunks,
            total_tokens=sum(c.tokens for c in chunks),
            created_at=datetime.utcnow().isoformat()
        )
        
        self.sessions[session_id] = session
        self.embeddings[session_id] = embed_array
        
        return session
    
    def search(
        self,
        session_id: str,
        query_embedding: list[float],
        top_k: int = 5,
        threshold: float = 0.0
    ) -> list[tuple[Chunk, float]]:
        """
        Search for similar chunks using cosine similarity.
        Returns list of (chunk, score) tuples sorted by score descending.
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        embed_matrix = self.embeddings[session_id]
        
        # Normalize query embedding
        query_vec = np.array(query_embedding, dtype=np.float32)
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        
        # Compute cosine similarities
        similarities = embed_matrix @ query_vec
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= threshold:
                results.append((session.chunks[idx], score))
        
        return results
    
    def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session data by ID."""
        return self.sessions.get(session_id)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            del self.embeddings[session_id]
            return True
        return False
    
    def list_sessions(self) -> list[str]:
        """List all session IDs."""
        return list(self.sessions.keys())


class DocumentProcessor:
    """
    Main class for document processing pipeline.
    Combines chunking, embedding, and storage.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.chunker = TextChunker()
        self.embedder = EmbeddingService(api_key)
        self.store = VectorStore()
    
    def ingest(
        self,
        text: str,
        doc_type: DocType = DocType.GENERAL,
        session_id: Optional[str] = None,
        title: Optional[str] = None
    ) -> SessionData:
        """
        Ingest a document: chunk, embed, and store.
        Returns the session data.
        """
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())[:8]
        
        # Chunk the text
        chunks = self.chunker.chunk_text(text)
        
        if not chunks:
            raise ValueError("Document is empty or could not be chunked")
        
        # Generate embeddings for all chunks
        chunk_texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_texts(chunk_texts)
        
        # Store in vector store
        session = self.store.add_session(
            session_id=session_id,
            doc_type=doc_type,
            original_text=text,
            chunks=chunks,
            embeddings=embeddings,
            title=title
        )
        
        return session
    
    def retrieve(
        self,
        session_id: str,
        query: str,
        top_k: int = 5
    ) -> list[tuple[Chunk, float]]:
        """
        Retrieve relevant chunks for a query.
        Returns list of (chunk, similarity_score) tuples.
        """
        # Embed the query
        query_embedding = self.embedder.embed_single(query)
        
        # Search
        return self.store.search(session_id, query_embedding, top_k=top_k)
    
    def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session by ID."""
        return self.store.get_session(session_id)


# Global instance (initialized lazily)
_processor: Optional[DocumentProcessor] = None


def get_processor() -> DocumentProcessor:
    """Get or create the global document processor."""
    global _processor
    if _processor is None:
        _processor = DocumentProcessor()
    return _processor
