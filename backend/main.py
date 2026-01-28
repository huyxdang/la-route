"""
Le-Route FastAPI Application
Enterprise Document Q&A with Intelligent Model Routing
"""

import os
import re
import time
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from mistralai import Mistral

from .models import (
    IngestRequest, IngestResponse,
    AskRequest, AskResponse,
    ChunkInfo, RoutingInfo,
    HealthResponse, SessionInfo,
    RiskLevel
)
from .embeddings import DocumentProcessor, get_processor
from .router import get_router, estimate_cost, RuleBasedRouter
from .prompts import SYSTEM_PROMPT, build_qa_prompt

# Load environment variables
load_dotenv()

# Global instances
processor: Optional[DocumentProcessor] = None
router = None
mistral_client: Optional[Mistral] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    global processor, router, mistral_client
    
    print("🚀 Starting Le-Route...")
    
    # Initialize document processor
    processor = get_processor()
    print("✓ Document processor initialized")
    
    # Initialize router
    router = get_router()
    print(f"✓ Router initialized: {router.name}")
    
    # Initialize Mistral client
    api_key = os.getenv("MISTRAL_API_KEY")
    if api_key:
        mistral_client = Mistral(api_key=api_key)
        print("✓ Mistral client initialized")
    else:
        print("⚠ MISTRAL_API_KEY not set")
    
    yield
    
    print("👋 Shutting down Le-Route...")


# Create FastAPI app
app = FastAPI(
    title="Le-Route",
    description="Enterprise Document Q&A with Intelligent Model Routing",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    from .router import MLPRouter
    
    return HealthResponse(
        status="healthy",
        router_type=router.name if router else "none",
        mlp_loaded=isinstance(router, MLPRouter) and router.loaded if router else False
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest):
    """
    Ingest a document: chunk, embed, and store.
    """
    if not processor:
        raise HTTPException(status_code=503, detail="Document processor not initialized")
    
    try:
        session = processor.ingest(
            text=request.text,
            doc_type=request.doc_type,
            session_id=request.session_id,
            title=request.title
        )
        
        return IngestResponse(
            session_id=session.session_id,
            chunks_created=len(session.chunks),
            total_tokens=session.total_tokens,
            doc_type=session.doc_type,
            status="ready"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Answer a question about an ingested document.
    Routes to optimal model and returns answer with citations.
    """
    if not processor or not mistral_client:
        raise HTTPException(status_code=503, detail="Service not fully initialized")
    
    start_time = time.time()
    
    # Get session
    session = processor.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {request.session_id} not found")
    
    # Retrieve relevant chunks
    results = processor.retrieve(request.session_id, request.question, top_k=5)
    
    if not results:
        # No relevant chunks found
        return AskResponse(
            answer="I cannot find relevant information in the document to answer this question.",
            citations=[],
            routing=RoutingInfo(
                model_used="none",
                routing_reason="No relevant chunks found",
                confidence=0.0,
                risk_level=RiskLevel.LOW,
                cost_estimate_usd=0.0,
                latency_ms=int((time.time() - start_time) * 1000)
            ),
            abstained=True
        )
    
    # Get similarities for routing
    top_similarities = [score for _, score in results]
    
    # Make routing decision
    if isinstance(router, RuleBasedRouter):
        decision = router.route(
            query=request.question,
            top_similarity=max(top_similarities),
            num_relevant_chunks=sum(1 for s in top_similarities if s > 0.5),
            doc_type=session.doc_type.value
        )
    else:
        # MLP router needs query embedding
        query_embedding = processor.embedder.embed_single(request.question)
        decision = router.route(
            query=request.question,
            query_embedding=query_embedding,
            top_similarities=top_similarities,
            query_token_count=len(request.question.split()),
            doc_token_count=session.total_tokens,
            doc_type=session.doc_type.value
        )
    
    # Build prompt with retrieved chunks
    chunks_for_prompt = [
        (chunk.text, score, chunk.chunk_id)
        for chunk, score in results
    ]
    user_prompt = build_qa_prompt(request.question, chunks_for_prompt)
    
    # Generate answer with selected model
    try:
        response = mistral_client.chat.complete(
            model=decision.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=1024
        )
        
        answer = response.choices[0].message.content
        
        # Extract usage for cost calculation
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model generation failed: {str(e)}")
    
    # Build citations from answer
    citations = []
    citation_pattern = r'\[(\d+)\]'
    cited_ids = set(int(m) for m in re.findall(citation_pattern, answer))
    
    for chunk, score in results:
        # Adjust for 1-based citation indexing in answer
        citation_num = chunk.chunk_id + 1
        
        if citation_num in cited_ids:
            # Find highlight position (simple heuristic: first sentence)
            first_sentence_end = chunk.text.find('. ')
            if first_sentence_end == -1:
                first_sentence_end = min(100, len(chunk.text))
            else:
                first_sentence_end += 1
            
            citations.append(ChunkInfo(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                relevance_score=score,
                highlight_start=0,
                highlight_end=first_sentence_end
            ))
    
    # Check for abstention
    abstained = "cannot find" in answer.lower() or "not in the document" in answer.lower()
    
    # Calculate latency and cost
    latency_ms = int((time.time() - start_time) * 1000)
    cost = estimate_cost(decision.model, input_tokens, output_tokens)
    
    return AskResponse(
        answer=answer,
        citations=citations,
        routing=RoutingInfo(
            model_used=decision.model,
            routing_reason=decision.reason,
            confidence=decision.confidence,
            risk_level=decision.risk_level,
            cost_estimate_usd=cost,
            latency_ms=latency_ms
        ),
        abstained=abstained
    )


@app.get("/session/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str):
    """Get information about a session."""
    if not processor:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    session = processor.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    return SessionInfo(
        session_id=session.session_id,
        doc_type=session.doc_type,
        title=session.title,
        chunk_count=len(session.chunks),
        total_tokens=session.total_tokens,
        created_at=session.created_at
    )


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    if not processor:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    if processor.store.delete_session(session_id):
        return {"status": "deleted", "session_id": session_id}
    else:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")


@app.get("/sessions")
async def list_sessions():
    """List all active sessions."""
    if not processor:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return {"sessions": processor.store.list_sessions()}


# Run with: uvicorn backend.main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
