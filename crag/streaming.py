"""
Streaming Pipeline for CRAG with Latency Tracking.

Provides streaming execution of the CRAG pipeline with:
- Status updates for each pipeline step with latency metrics
- Token-by-token streaming during answer generation
- Structured events for frontend consumption

Usage:
    from crag.streaming import PipelineStreamer, StreamEvent
    
    streamer = PipelineStreamer(retriever=retriever)
    
    for event in streamer.stream("What is GPT-4?", session_id="abc"):
        if event.event == "status":
            print(f"[{event.data['latency_ms']}ms] {event.data['step']}")
        elif event.event == "token":
            print(event.data['content'], end="", flush=True)
        elif event.event == "done":
            print(f"\\nTotal: {event.data['total_ms']}ms")
"""

import os
import time
import json
from dataclasses import dataclass, field, asdict
from typing import Generator, List, Optional, Literal, Dict, Any, Union
from mistralai import Mistral

from .graph_state import GraphState, Message
from .retrieval import Document, HybridRetriever, PineconeHybridRetriever, InMemoryHybridRetriever
from .graders import (
    QueryRewriterAndRouter,
    DocumentGrader,
    DocumentReranker,
    GenerationGrader,
)
from .config import (
    GENERATION_MODEL,
    CONVERSATIONAL_MODEL,
    CONVERSATIONAL_SYSTEM_PROMPT,
    GENERATION_SYSTEM_PROMPT,
    GENERATION_SIMPLE_PROMPT,
    MAX_GENERATION_TOKENS,
)
from .citations import CitationExtractor, CitedResponse, Citation

# Type alias for retrievers
RetrieverType = Union[HybridRetriever, PineconeHybridRetriever, InMemoryHybridRetriever]


# ============== Event Types ==============

@dataclass
class StreamEvent:
    """
    A streaming event from the CRAG pipeline.
    
    Event types:
    - "status": Pipeline step completed (includes latency_ms)
    - "token": A token from the generation stream
    - "citations": Structured citations extracted from response
    - "error": An error occurred
    - "done": Pipeline completed
    
    Attributes:
        event: The event type
        data: Event-specific data
    """
    event: Literal["status", "token", "citations", "error", "done"]
    data: Dict[str, Any]
    
    def to_sse(self) -> str:
        """Format as Server-Sent Event string."""
        return f"data: {json.dumps({'event': self.event, 'data': self.data})}\n\n"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {"event": self.event, "data": self.data}
    
    def __repr__(self) -> str:
        if self.event == "token":
            content = self.data.get("content", "")[:20]
            return f"StreamEvent(token, '{content}...')"
        return f"StreamEvent({self.event}, {self.data})"


def status_event(
    step: str,
    latency_ms: int,
    details: Optional[Dict[str, Any]] = None
) -> StreamEvent:
    """Create a status event with latency."""
    data = {"step": step, "latency_ms": latency_ms}
    if details:
        data["details"] = details
    return StreamEvent(event="status", data=data)


def token_event(content: str) -> StreamEvent:
    """Create a token event."""
    return StreamEvent(event="token", data={"content": content})


def error_event(message: str, step: Optional[str] = None) -> StreamEvent:
    """Create an error event."""
    data = {"message": message}
    if step:
        data["step"] = step
    return StreamEvent(event="error", data=data)


def done_event(
    session_id: str,
    total_ms: int,
    grade: Optional[str] = None,
    generation: Optional[str] = None,
    citations: Optional[List[Dict]] = None,
) -> StreamEvent:
    """Create a completion event."""
    data = {
        "session_id": session_id,
        "total_ms": total_ms,
    }
    if grade:
        data["grade"] = grade
    if generation:
        data["generation"] = generation
    if citations:
        data["citations"] = citations
    return StreamEvent(event="done", data=data)


def citation_event(citations: List[Dict]) -> StreamEvent:
    """Create a citations event with structured source information."""
    return StreamEvent(event="citations", data={"citations": citations})


# ============== Timer Utility ==============

class Timer:
    """Simple timer for measuring latency."""
    
    def __init__(self):
        self._start: Optional[float] = None
        self._total_start: Optional[float] = None
    
    def start(self):
        """Start/reset the step timer."""
        self._start = time.perf_counter()
        if self._total_start is None:
            self._total_start = self._start
    
    def elapsed_ms(self) -> int:
        """Get elapsed time in milliseconds since last start()."""
        if self._start is None:
            return 0
        return int((time.perf_counter() - self._start) * 1000)
    
    def total_ms(self) -> int:
        """Get total elapsed time since first start()."""
        if self._total_start is None:
            return 0
        return int((time.perf_counter() - self._total_start) * 1000)


# ============== Pipeline Streamer ==============

class PipelineStreamer:
    """
    Streaming CRAG pipeline with latency tracking.
    
    Executes the CRAG pipeline step-by-step, yielding StreamEvent objects
    that include latency metrics for each step and token-by-token generation.
    
    Usage:
        streamer = PipelineStreamer(
            retriever=retriever,
            retrieval_top_k=50,
            rerank_top_k=10,
        )
        
        for event in streamer.stream("What is GPT-4?"):
            print(event)
    """
    
    def __init__(
        self,
        retriever: RetrieverType,
        query_processor: Optional[QueryRewriterAndRouter] = None,
        doc_grader: Optional[DocumentGrader] = None,
        doc_reranker: Optional[DocumentReranker] = None,
        gen_grader: Optional[GenerationGrader] = None,
        web_search_tool=None,
        retrieval_top_k: int = 50,
        rerank_top_k: int = 10,
        generation_model: str = GENERATION_MODEL,
        use_citations: bool = True,
    ):
        """
        Initialize the pipeline streamer.
        
        Args:
            retriever: Required retriever instance
            query_processor: Optional QueryRewriterAndRouter (created if None)
            doc_grader: Optional DocumentGrader (created if None)
            doc_reranker: Optional DocumentReranker (created if None)
            gen_grader: Optional GenerationGrader (created if None)
            web_search_tool: Optional web search tool
            retrieval_top_k: Number of docs to retrieve
            rerank_top_k: Number of docs after reranking
            generation_model: Model for generation (default: mistral-large-latest)
            use_citations: Whether to use structured citations (default: True)
        """
        self.retriever = retriever
        self.query_processor = query_processor or QueryRewriterAndRouter()
        self.doc_grader = doc_grader or DocumentGrader()
        self.doc_reranker = doc_reranker or DocumentReranker()
        self.gen_grader = gen_grader or GenerationGrader()
        self.web_search_tool = web_search_tool
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_k = rerank_top_k
        self.generation_model = generation_model
        self.use_citations = use_citations
        self._citation_extractor = None
        
        # Lazy-init Mistral client
        self._mistral_client = None
    
    def _get_mistral(self) -> Mistral:
        """Get or create Mistral client."""
        if self._mistral_client is None:
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("MISTRAL_API_KEY not found in environment")
            self._mistral_client = Mistral(api_key=api_key)
        return self._mistral_client
    
    def _init_web_search(self):
        """Initialize web search tool if not provided."""
        if self.web_search_tool is None:
            try:
                from langchain_tavily import TavilySearch
                self.web_search_tool = TavilySearch(max_results=3)
            except ImportError:
                pass
    
    def _get_citation_extractor(self) -> CitationExtractor:
        """Get or create citation extractor."""
        if self._citation_extractor is None:
            self._citation_extractor = CitationExtractor(model=self.generation_model)
        return self._citation_extractor
    
    def stream(
        self,
        question: str,
        history: Optional[List[Message]] = None,
        session_id: Optional[str] = None,
    ) -> Generator[StreamEvent, None, None]:
        """
        Stream the CRAG pipeline execution.
        
        Yields StreamEvent objects for:
        - Each pipeline step (with latency_ms)
        - Each token during generation
        - Completion with total time
        
        Args:
            question: The user's question
            history: Optional conversation history
            session_id: Optional session ID for the done event
            
        Yields:
            StreamEvent objects
        """
        timer = Timer()
        history = history or []
        session_id = session_id or "unknown"
        
        # Track state through the pipeline
        state = {
            "question": question,
            "rewritten_query": None,
            "documents": [],
            "generation": "",
            "route_decision": None,
            "web_search_done": False,
            "generation_grade": None,
            "citations": [],
        }
        
        try:
            # Step 1: Process query (rewrite + route in one LLM call)
            timer.start()
            result = self.query_processor.process(question, history)
            state["rewritten_query"] = result.query
            state["route_decision"] = result.route
            query_for_search = result.query
            
            yield status_event(
                step="processing",
                latency_ms=timer.elapsed_ms(),
                details={
                    "original": question,
                    "rewritten": result.query,
                    "route": result.route,
                    "reasoning": result.reasoning[:200],
                }
            )
            
            # Step 2: Handle conversational route
            if result.route == "conversational":
                # Conversational response - skip retrieval entirely
                yield from self._generate_conversational(question, state, timer)
                # Done event for conversational
                yield done_event(
                    session_id=session_id,
                    total_ms=timer.total_ms(),
                    grade="conversational",
                    generation=state.get("generation"),
                    citations=[]
                )
                return  # Exit early, no need for grading
            
            # Step 3: Route to appropriate handler
            if result.route == "web_search":
                # Direct to web search
                yield from self._do_web_search(state, timer)
            else:
                # Vectorstore path
                timer.start()
                documents = self.retriever.retrieve(query_for_search, top_k=self.retrieval_top_k)
                state["documents"] = documents
                yield status_event(
                    step="retrieving",
                    latency_ms=timer.elapsed_ms(),
                    details={"count": len(documents)}
                )
                
                # Step 4: Rerank
                if documents:
                    timer.start()
                    reranked = self.doc_reranker.rerank(documents, query_for_search, top_k=self.rerank_top_k)
                    state["documents"] = reranked
                    yield status_event(
                        step="reranking",
                        latency_ms=timer.elapsed_ms(),
                        details={"before": len(documents), "after": len(reranked)}
                    )
                
                # Step 5: Grade documents
                if state["documents"]:
                    timer.start()
                    filtered_docs, needs_web = self.doc_grader.grade_documents(state["documents"], question)
                    state["documents"] = filtered_docs
                    yield status_event(
                        step="grading_docs",
                        latency_ms=timer.elapsed_ms(),
                        details={"kept": len(filtered_docs), "needs_web_search": needs_web}
                    )
                    
                    # If no relevant docs, fall back to web search
                    if needs_web and not state["web_search_done"]:
                        yield from self._do_web_search(state, timer)
            
            # Step 6: Generate answer (streaming with citations)
            yield from self._generate_streaming(state, timer)
            
            # Step 6b: Extract and yield citations if enabled
            if self.use_citations and state["generation"] and state["documents"]:
                timer.start()
                extractor = self._get_citation_extractor()
                cited_response = extractor.extract_citations_from_text(
                    state["generation"],
                    state["documents"]
                )
                state["citations"] = [c.to_dict() for c in cited_response.citations]
                state["generation"] = cited_response.answer  # Clean answer without JSON
                
                if state["citations"]:
                    yield citation_event(state["citations"])
                    yield status_event(
                        step="extracting_citations",
                        latency_ms=timer.elapsed_ms(),
                        details={"count": len(state["citations"])}
                    )
            
            # Step 7: Grade generation
            timer.start()
            if state["documents"] and state["generation"]:
                doc_texts = "\n\n---\n\n".join(
                    doc.content if hasattr(doc, 'content') else str(doc)
                    for doc in state["documents"]
                )
                grade = self.gen_grader.grade(doc_texts, state["generation"], question)
                state["generation_grade"] = grade
                yield status_event(
                    step="grading_gen",
                    latency_ms=timer.elapsed_ms(),
                    details={"grade": grade}
                )
            else:
                yield status_event(
                    step="grading_gen",
                    latency_ms=timer.elapsed_ms(),
                    details={"skipped": True, "reason": "no docs or generation"}
                )
            
            # Done
            yield done_event(
                session_id=session_id,
                total_ms=timer.total_ms(),
                grade=state.get("generation_grade"),
                generation=state["generation"],
                citations=state.get("citations"),
            )
            
        except Exception as e:
            yield error_event(message=str(e), step="unknown")
            yield done_event(
                session_id=session_id,
                total_ms=timer.total_ms(),
                grade="error",
            )
    
    def _do_web_search(
        self,
        state: dict,
        timer: Timer,
    ) -> Generator[StreamEvent, None, None]:
        """Perform web search and yield status event."""
        timer.start()
        self._init_web_search()
        
        if self.web_search_tool is None:
            yield status_event(
                step="web_search",
                latency_ms=timer.elapsed_ms(),
                details={"skipped": True, "reason": "no web search tool"}
            )
            return
        
        try:
            results = self.web_search_tool.invoke({"query": state["question"]})

            # Parse results into documents
            web_docs = []
            # TavilySearch returns a dict with a "results" key
            if isinstance(results, dict):
                for r in results.get("results", []):
                    content = r.get("content", "") or r.get("snippet", "")
                    url = r.get("url", "")
                    title = r.get("title", "")
                    if content:
                        web_docs.append(Document(
                            content=content,
                            metadata={"source": "web", "url": url, "title": title},
                            source="web",
                        ))
            elif isinstance(results, str):
                web_docs.append(Document(content=results, metadata={"source": "web"}, source="web"))
            elif isinstance(results, list):
                for r in results:
                    if isinstance(r, dict):
                        content = r.get("content", "") or r.get("snippet", "") or str(r)
                    else:
                        content = str(r)
                    if content:
                        web_docs.append(Document(content=content, metadata={"source": "web"}, source="web"))
            
            state["documents"].extend(web_docs)
            state["web_search_done"] = True
            
            yield status_event(
                step="web_search",
                latency_ms=timer.elapsed_ms(),
                details={"count": len(web_docs)}
            )
        except Exception as e:
            yield status_event(
                step="web_search",
                latency_ms=timer.elapsed_ms(),
                details={"error": str(e)}
            )
            state["web_search_done"] = True
    
    def _generate_conversational(
        self,
        question: str,
        state: dict,
        timer: Timer,
    ) -> Generator[StreamEvent, None, None]:
        """
        Generate a friendly conversational response using ministral-3b.
        
        Used for greetings, off-topic queries, and capability questions.
        Skips retrieval and citation extraction entirely.
        """
        timer.start()
        
        yield status_event(
            step="generating",
            latency_ms=0,
            details={"status": "starting", "mode": "conversational"}
        )
        
        client = self._get_mistral()
        full_response = ""
        token_count = 0
        
        try:
            stream = client.chat.stream(
                model=CONVERSATIONAL_MODEL,
                messages=[
                    {"role": "system", "content": CONVERSATIONAL_SYSTEM_PROMPT},
                    {"role": "user", "content": question}
                ],
                temperature=0.7,
                max_tokens=500,
            )
            
            for chunk in stream:
                if chunk.data.choices and chunk.data.choices[0].delta.content:
                    token = chunk.data.choices[0].delta.content
                    full_response += token
                    token_count += 1
                    yield token_event(token)
            
            state["generation"] = full_response
            
            yield status_event(
                step="generating",
                latency_ms=timer.elapsed_ms(),
                details={"tokens": token_count, "mode": "conversational"}
            )
            
        except Exception as e:
            yield error_event(f"Conversational generation failed: {str(e)}", step="generating")
            state["generation"] = "I'm having trouble responding right now. How can I help you with NeurIPS 2025 research?"
    
    def _generate_streaming(
        self,
        state: dict,
        timer: Timer,
    ) -> Generator[StreamEvent, None, None]:
        """Generate answer with token streaming."""
        timer.start()
        
        documents = state["documents"]
        question = state["question"]
        
        # If no documents, generate a "no context" response
        if not documents:
            state["generation"] = "I don't have enough information to answer this question based on the available documents."
            yield status_event(
                step="generating",
                latency_ms=timer.elapsed_ms(),
                details={"tokens": 0, "reason": "no documents"}
            )
            return
        
        # Format context with titles if available
        context_parts = []
        for i, doc in enumerate(documents):
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            title = metadata.get("title", "")
            content = doc.content if hasattr(doc, 'content') else str(doc)
            
            header = f"[Document {i+1}]"
            if title:
                header += f" - {title}"
            
            context_parts.append(f"{header}\n{content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Use citation-aware prompt if enabled
        if self.use_citations:
            system_prompt = GENERATION_SYSTEM_PROMPT
        else:
            system_prompt = GENERATION_SIMPLE_PROMPT

        context_prompt = f"""## Retrieved Documents:
{context}

## User Question:
{question}

## Your Answer:"""

        # Yield status that generation is starting
        yield status_event(
            step="generating",
            latency_ms=0,
            details={"status": "starting", "doc_count": len(documents)}
        )
        
        # Stream generation
        client = self._get_mistral()
        full_response = ""
        token_count = 0
        
        try:
            stream = client.chat.stream(
                model=self.generation_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context_prompt}
                ],
                temperature=0.8,
                max_tokens=MAX_GENERATION_TOKENS,
            )
            
            for chunk in stream:
                if chunk.data.choices and chunk.data.choices[0].delta.content:
                    token = chunk.data.choices[0].delta.content
                    full_response += token
                    token_count += 1
                    yield token_event(token)
            
            state["generation"] = full_response.strip()
            
            # Final generation status with total time
            yield status_event(
                step="generating",
                latency_ms=timer.elapsed_ms(),
                details={"status": "complete", "tokens": token_count, "chars": len(full_response)}
            )
            
        except Exception as e:
            yield error_event(message=f"Generation failed: {str(e)}", step="generating")
            state["generation"] = ""


# ============== Convenience Function ==============

def stream_crag_with_events(
    question: str,
    retriever: RetrieverType,
    history: Optional[List[Message]] = None,
    session_id: Optional[str] = None,
    **kwargs
) -> Generator[StreamEvent, None, None]:
    """
    Stream CRAG pipeline execution with events.
    
    Convenience wrapper around PipelineStreamer.
    
    Args:
        question: The user's question
        retriever: Required retriever instance
        history: Optional conversation history
        session_id: Optional session ID
        **kwargs: Additional arguments for PipelineStreamer
        
    Yields:
        StreamEvent objects for each pipeline step and token
        
    Example:
        for event in stream_crag_with_events("What is GPT-4?", retriever):
            if event.event == "token":
                print(event.data["content"], end="")
    """
    streamer = PipelineStreamer(retriever=retriever, **kwargs)
    yield from streamer.stream(question, history=history, session_id=session_id)
