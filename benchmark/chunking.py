"""
Le-Route: Semantic chunking and retrieval for large documents.

Uses Mistral's tokenizer (mistral-common) for accurate token counting 
and mistral-embed for semantic chunking and retrieval.
"""

import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from mistralai import Mistral
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")


# ============== Token Counting ==============

# Initialize Mistral tokenizer (v7 is the latest tokenizer for Mistral models)
_tokenizer = None

def get_tokenizer() -> MistralTokenizer:
    """Get or initialize the Mistral tokenizer (lazy loading)."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = MistralTokenizer.v7()
    return _tokenizer


def count_tokens(text: str) -> int:
    """
    Count tokens in text using Mistral's tokenizer.
    
    Uses mistral-common for accurate token counting.
    """
    tokenizer = get_tokenizer()
    tokens = tokenizer.instruct_tokenizer.tokenizer.encode(text, bos=False, eos=False)
    return len(tokens)


def estimate_doc_tokens(doc) -> int:
    """
    Count total tokens for an ExtractedDocument.
    
    Args:
        doc: ExtractedDocument from pdf_extract
        
    Returns:
        Token count
    """
    total_text = ""
    for page in doc.pages:
        total_text += page.text
    return count_tokens(total_text)


# ============== Semantic Chunking ==============

@dataclass
class Chunk:
    """A semantic chunk of document text."""
    text: str
    page_nums: list[int]  # Pages this chunk spans
    embedding: Optional[np.ndarray] = None
    token_count: int = 0


@dataclass
class ChunkedDocument:
    """Document split into semantic chunks."""
    chunks: list[Chunk]
    total_tokens: int
    total_pages: int


class SemanticChunker:
    """
    Splits documents into semantically coherent chunks using embeddings.
    
    Strategy:
    1. Split document into paragraphs by page
    2. Embed each paragraph with mistral-embed
    3. Find semantic boundaries (where similarity drops)
    4. Merge adjacent similar paragraphs into target-sized chunks
    """
    
    def __init__(
        self,
        target_chunk_tokens: int = 1500,
        min_chunk_tokens: int = 200,
        similarity_threshold: float = 0.7
    ):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment")
        
        self.client = Mistral(api_key=api_key)
        self.target_chunk_tokens = target_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.similarity_threshold = similarity_threshold
    
    def _get_embeddings(self, texts: list[str], batch_size: int = 20) -> list[np.ndarray]:
        """Get embeddings for a list of texts using mistral-embed."""
        if not texts:
            return []
        
        all_embeddings = []
        
        # Truncate very long texts to avoid token limits (mistral-embed limit ~8K tokens)
        max_chars = 8000  # ~2000 tokens
        truncated_texts = [t[:max_chars] if len(t) > max_chars else t for t in texts]
        
        # Process in batches to avoid API token limits
        for i in range(0, len(truncated_texts), batch_size):
            batch = truncated_texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model="mistral-embed",
                inputs=batch
            )
            all_embeddings.extend([np.array(item.embedding) for item in response.data])
        
        return all_embeddings
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _split_into_paragraphs(self, doc) -> list[dict]:
        """
        Split document into paragraphs with page numbers.
        
        Returns list of {"text": str, "page_num": int}
        """
        paragraphs = []
        
        for page in doc.pages:
            # Split page text by double newlines (paragraph breaks)
            page_text = page.text.strip()
            if not page_text:
                continue
            
            # Split into paragraphs
            para_texts = [p.strip() for p in page_text.split('\n\n') if p.strip()]
            
            for para in para_texts:
                # Skip very short paragraphs (likely headers, page numbers)
                if len(para) < 50:
                    continue
                
                paragraphs.append({
                    "text": para,
                    "page_num": page.page_num,
                    "tokens": count_tokens(para)
                })
        
        return paragraphs
    
    def chunk_document(self, doc) -> ChunkedDocument:
        """
        Split document into semantic chunks.
        
        Args:
            doc: ExtractedDocument from pdf_extract
            
        Returns:
            ChunkedDocument with embedded chunks
        """
        paragraphs = self._split_into_paragraphs(doc)
        
        if not paragraphs:
            return ChunkedDocument(chunks=[], total_tokens=0, total_pages=doc.total_pages)
        
        # Get embeddings for all paragraphs
        para_texts = [p["text"] for p in paragraphs]
        embeddings = self._get_embeddings(para_texts)
        
        # Find semantic boundaries
        boundaries = [0]  # Start of first chunk
        for i in range(1, len(embeddings)):
            sim = self._cosine_similarity(embeddings[i-1], embeddings[i])
            if sim < self.similarity_threshold:
                boundaries.append(i)
        boundaries.append(len(paragraphs))  # End of last chunk
        
        # Create chunks from boundaries, respecting size limits
        chunks = []
        current_text = ""
        current_pages = set()
        current_tokens = 0
        
        for i, para in enumerate(paragraphs):
            # Check if we should start a new chunk
            is_boundary = i in boundaries[1:-1]  # Exclude first and last
            would_exceed = current_tokens + para["tokens"] > self.target_chunk_tokens
            
            if (is_boundary or would_exceed) and current_tokens >= self.min_chunk_tokens:
                # Save current chunk
                chunks.append(Chunk(
                    text=current_text.strip(),
                    page_nums=sorted(current_pages),
                    token_count=current_tokens
                ))
                current_text = ""
                current_pages = set()
                current_tokens = 0
            
            # Add paragraph to current chunk
            current_text += para["text"] + "\n\n"
            current_pages.add(para["page_num"])
            current_tokens += para["tokens"]
        
        # Don't forget the last chunk
        if current_text.strip():
            chunks.append(Chunk(
                text=current_text.strip(),
                page_nums=sorted(current_pages),
                token_count=current_tokens
            ))
        
        # Embed all chunks
        chunk_texts = [c.text for c in chunks]
        chunk_embeddings = self._get_embeddings(chunk_texts)
        for chunk, emb in zip(chunks, chunk_embeddings):
            chunk.embedding = emb
        
        total_tokens = sum(c.token_count for c in chunks)
        
        return ChunkedDocument(
            chunks=chunks,
            total_tokens=total_tokens,
            total_pages=doc.total_pages
        )


# ============== Chunk Retrieval ==============

class ChunkRetriever:
    """Retrieves relevant chunks for a question using embedding similarity."""
    
    def __init__(self, chunked_doc: ChunkedDocument, top_k: int = 5):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment")
        
        self.client = Mistral(api_key=api_key)
        self.chunked_doc = chunked_doc
        self.top_k = top_k
        
        # Pre-compute embedding matrix for fast retrieval
        self._embeddings = np.array([c.embedding for c in chunked_doc.chunks])
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text."""
        response = self.client.embeddings.create(
            model="mistral-embed",
            inputs=[text]
        )
        return np.array(response.data[0].embedding)
    
    def retrieve(self, question: str) -> tuple[list[Chunk], set[int]]:
        """
        Retrieve top-K chunks most relevant to the question.
        
        Args:
            question: The user's question
            
        Returns:
            Tuple of (list of relevant chunks, set of page numbers covered)
        """
        if len(self.chunked_doc.chunks) == 0:
            return [], set()
        
        # Embed the question
        q_embedding = self._get_embedding(question)
        
        # Compute similarities
        similarities = np.dot(self._embeddings, q_embedding) / (
            np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(q_embedding)
        )
        
        # Get top-K indices
        top_indices = np.argsort(similarities)[-self.top_k:][::-1]
        
        # Collect chunks and pages
        retrieved_chunks = [self.chunked_doc.chunks[i] for i in top_indices]
        retrieved_pages = set()
        for chunk in retrieved_chunks:
            retrieved_pages.update(chunk.page_nums)
        
        return retrieved_chunks, retrieved_pages


# ============== Content Builder for RAG ==============

def build_rag_content(
    chunks: list[Chunk],
    doc,  # ExtractedDocument
    retrieved_pages: set[int],
    max_images: int = 8,
    include_images: bool = True
) -> list[dict]:
    """
    Build Mistral API content from retrieved chunks and page-associated images.
    
    Args:
        chunks: Retrieved chunks
        doc: Original ExtractedDocument (for images)
        retrieved_pages: Set of page numbers covered by chunks
        max_images: Maximum images to include
        include_images: Whether to include images (default: True)
        
    Returns:
        Content list for Mistral chat API
    """
    content = []
    
    # 1. Add retrieved text chunks with page references
    text_parts = []
    for i, chunk in enumerate(chunks):
        pages_str = ", ".join(str(p) for p in chunk.page_nums)
        text_parts.append(f"[Relevant excerpt from page(s) {pages_str}]\n{chunk.text}")
    
    if text_parts:
        content.append({
            "type": "text",
            "text": "## Retrieved Document Excerpts:\n\n" + "\n\n---\n\n".join(text_parts)
        })
    
    # 2. Add images from retrieved pages only (if enabled)
    if include_images:
        image_count = 0
        for page in doc.pages:
            if page.page_num not in retrieved_pages:
                continue
            
            for img in page.images:
                if image_count >= max_images:
                    break
                
                content.append({
                    "type": "text",
                    "text": f"[Figure from page {img.page_num}]"
                })
                
                mime_type = f"image/{img.image_type}"
                if img.image_type == "jpg":
                    mime_type = "image/jpeg"
                
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{img.image_b64}"
                    }
                })
                image_count += 1
            
            if image_count >= max_images:
                break
    
    return content


# ============== Example Usage ==============

if __name__ == "__main__":
    import sys
    from pdf_extract import extract_pdf
    
    if len(sys.argv) < 2:
        print("Usage: python chunking.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    print(f"Extracting: {pdf_path}")
    
    # Extract document
    doc = extract_pdf(pdf_path)
    print(f"Document: {doc.total_pages} pages, ~{estimate_doc_tokens(doc)} tokens")
    
    # Chunk document
    print("\nChunking document...")
    chunker = SemanticChunker()
    chunked = chunker.chunk_document(doc)
    
    print(f"\nCreated {len(chunked.chunks)} chunks:")
    for i, chunk in enumerate(chunked.chunks):
        print(f"  Chunk {i+1}: {chunk.token_count} tokens, pages {chunk.page_nums}")
    
    # Test retrieval
    if len(sys.argv) > 2:
        question = sys.argv[2]
        print(f"\nRetrieving for question: {question}")
        
        retriever = ChunkRetriever(chunked)
        chunks, pages = retriever.retrieve(question)
        
        print(f"Retrieved {len(chunks)} chunks from pages: {sorted(pages)}")
        for i, chunk in enumerate(chunks):
            print(f"\n--- Chunk {i+1} (pages {chunk.page_nums}) ---")
            print(chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text)
