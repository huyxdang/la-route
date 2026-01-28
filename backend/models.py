"""
Pydantic models for Le-Route API request/response schemas.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class DocType(str, Enum):
    """Document type classification."""
    POLICY = "policy"
    CONTRACT = "contract"
    LEGAL = "legal"
    TECHNICAL = "technical"
    GENERAL = "general"


class RiskLevel(str, Enum):
    """Risk level for routing decisions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ============== Ingest Endpoint ==============

class IngestRequest(BaseModel):
    """Request body for document ingestion."""
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for the document. Auto-generated if not provided."
    )
    text: str = Field(
        ...,
        description="Full document text to ingest.",
        min_length=1
    )
    doc_type: DocType = Field(
        default=DocType.GENERAL,
        description="Type of document for routing optimization."
    )
    title: Optional[str] = Field(
        default=None,
        description="Optional title for the document."
    )


class IngestResponse(BaseModel):
    """Response body for document ingestion."""
    session_id: str = Field(..., description="Session ID for future queries.")
    chunks_created: int = Field(..., description="Number of chunks created.")
    total_tokens: int = Field(..., description="Total tokens in document.")
    doc_type: DocType = Field(..., description="Document type.")
    status: str = Field(default="ready", description="Ingestion status.")


# ============== Ask Endpoint ==============

class AskRequest(BaseModel):
    """Request body for asking questions."""
    session_id: str = Field(..., description="Session ID from ingestion.")
    question: str = Field(
        ...,
        description="Question to ask about the document.",
        min_length=1
    )


class ChunkInfo(BaseModel):
    """Information about a cited chunk."""
    chunk_id: int = Field(..., description="Chunk index in the document.")
    text: str = Field(..., description="Full chunk text.")
    relevance_score: float = Field(
        ...,
        description="Cosine similarity score (0-1).",
        ge=0.0,
        le=1.0
    )
    highlight_start: Optional[int] = Field(
        default=None,
        description="Character offset where relevant text starts."
    )
    highlight_end: Optional[int] = Field(
        default=None,
        description="Character offset where relevant text ends."
    )


class RoutingInfo(BaseModel):
    """Information about the routing decision."""
    model_used: str = Field(..., description="Model that was used.")
    routing_reason: str = Field(..., description="Why this model was chosen.")
    confidence: float = Field(
        ...,
        description="Router confidence (0-1).",
        ge=0.0,
        le=1.0
    )
    risk_level: RiskLevel = Field(..., description="Risk level of the query.")
    cost_estimate_usd: float = Field(
        ...,
        description="Estimated cost in USD.",
        ge=0.0
    )
    latency_ms: int = Field(..., description="Response latency in milliseconds.")


class AskResponse(BaseModel):
    """Response body for asking questions."""
    answer: str = Field(..., description="Answer with inline citations like [1], [2].")
    citations: list[ChunkInfo] = Field(
        default_factory=list,
        description="List of cited chunks."
    )
    routing: RoutingInfo = Field(..., description="Routing decision details.")
    abstained: bool = Field(
        default=False,
        description="Whether the model abstained from answering."
    )


# ============== Session Management ==============

class SessionInfo(BaseModel):
    """Information about a session."""
    session_id: str
    doc_type: DocType
    title: Optional[str]
    chunk_count: int
    total_tokens: int
    created_at: str


# ============== Internal Models ==============

class Chunk(BaseModel):
    """Internal representation of a document chunk."""
    chunk_id: int
    text: str
    tokens: int
    embedding: Optional[list[float]] = None
    start_char: int = Field(..., description="Start character position in original doc.")
    end_char: int = Field(..., description="End character position in original doc.")


class SessionData(BaseModel):
    """Internal session storage."""
    session_id: str
    doc_type: DocType
    title: Optional[str]
    original_text: str
    chunks: list[Chunk]
    total_tokens: int
    created_at: str


# ============== Health Check ==============

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    router_type: str = "rule"
    mlp_loaded: bool = False
