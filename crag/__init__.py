"""
Corrective RAG (CRAG) with Hybrid Search using LangGraph.

A stateful RAG workflow that:
1. Routes queries to vectorstore or web search
2. Performs hybrid retrieval via Pinecone (BM25 + Semantic)
3. Grades document relevance with Mistral Large
4. Falls back to web search if context is insufficient
5. Validates generation for hallucinations and usefulness

All LLM operations use Mistral Large (mistral-large-latest).
Vector storage uses Pinecone with persistent hybrid search.

Usage:
    >>> from crag import run_crag, PineconeHybridRetriever, Document
    >>> 
    >>> # Create retriever (connects to Pinecone index)
    >>> retriever = PineconeHybridRetriever(
    ...     index_name="my-crag-index",
    ...     namespace="my-project"
    ... )
    >>> 
    >>> # Add documents (one-time indexing)
    >>> docs = [
    ...     Document(content="GPT-4 achieved 86.4% on MMLU."),
    ...     Document(content="Claude 2 scored 78.5% on MMLU."),
    ... ]
    >>> retriever.add_documents(docs)
    >>> 
    >>> # Run CRAG (uses persistent Pinecone index)
    >>> result = run_crag("What is GPT-4's MMLU score?", retriever=retriever)
    >>> print(result["generation"])
"""

from .graph_state import GraphState
from .retrieval import (
    Document,
    HybridRetriever,
    PineconeHybridRetriever,
    InMemoryHybridRetriever,
)
from .graders import (
    QueryRouter,
    DocumentGrader,
    HallucinationGrader,
    AnswerGrader,
    GenerationGrader,
    DEFAULT_MODEL,
)
from .graph import (
    create_crag_graph,
    compile_crag_graph,
    run_crag,
    stream_crag,
)

__all__ = [
    # State
    "GraphState",
    # Retrieval
    "Document",
    "HybridRetriever",
    "PineconeHybridRetriever",
    "InMemoryHybridRetriever",
    # Graders
    "QueryRouter",
    "DocumentGrader",
    "HallucinationGrader",
    "AnswerGrader",
    "GenerationGrader",
    "DEFAULT_MODEL",
    # Graph
    "create_crag_graph",
    "compile_crag_graph",
    "run_crag",
    "stream_crag",
]
