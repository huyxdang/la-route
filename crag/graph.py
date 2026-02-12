"""
CRAG LangGraph Workflow.

Builds the stateful graph with nodes and conditional edges.
All LLM operations use Mistral Large.
Vector storage uses Pinecone with hybrid search.

Uses closure pattern for dependency injection - no global state.
"""

import os
from typing import List, Optional, Union
from langgraph.graph import StateGraph, END
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
    CONVERSATIONAL_MODEL,
    GENERATION_MODEL,
    CONVERSATIONAL_SYSTEM_PROMPT,
)

# Type alias for any retriever type
RetrieverType = Union[HybridRetriever, PineconeHybridRetriever, InMemoryHybridRetriever]


def create_crag_graph(
    retriever: RetrieverType,
    query_processor: Optional[QueryRewriterAndRouter] = None,
    doc_grader: Optional[DocumentGrader] = None,
    doc_reranker: Optional[DocumentReranker] = None,
    gen_grader: Optional[GenerationGrader] = None,
    web_search_tool = None,
    retrieval_top_k: int = 50,
    rerank_top_k: int = 10,
) -> StateGraph:
    """
    Create the CRAG workflow graph with dependency injection.
    
    All nodes are defined as closures that close over the provided dependencies,
    eliminating global state and making the graph testable.
    
    Graph Structure:
    ```
    START → process_query (rewrite + route in one LLM call)
                ↓
        ┌───────┼───────────────┐
        ↓       ↓               ↓
    convers. vectorstore    web_search
        ↓       ↓               │
       END  retrieve (50)       │
                ↓               │
            rerank (→10)        │
                ↓               │
            grade_documents     │
                ↓               │
                ├───────────────┤
                ↓               ↓
            (if irrelevant)  generate ←─┐
                ↓               ↓       │
            web_search     grade_gen    │
                ↓               ↓       │
                └──→ generate   ├───────┘ (if not useful & !web_search_done)
                                ↓
                               END (if useful or end_degraded)
    ```
    
    Args:
        retriever: Required HybridRetriever instance for vectorstore retrieval.
        query_processor: Optional QueryRewriterAndRouter (defaults to new instance)
        doc_grader: Optional DocumentGrader (defaults to new instance)
        doc_reranker: Optional DocumentReranker for reranking retrieved docs
        gen_grader: Optional GenerationGrader (defaults to new instance)
        web_search_tool: Optional web search tool (defaults to TavilySearch)
        retrieval_top_k: Number of docs to retrieve before reranking (default: 50)
        rerank_top_k: Number of docs to keep after reranking (default: 10)
                   
    Returns:
        StateGraph workflow (call .compile() to get executable)
    """
    # Default instantiation for optional dependencies (each uses its optimal model)
    query_processor = query_processor or QueryRewriterAndRouter()  # mistral-small (rewrite + route)
    doc_grader = doc_grader or DocumentGrader()   # mistral-large (context understanding)
    doc_reranker = doc_reranker or DocumentReranker()   # cohere/cross-encoder (reranking)
    gen_grader = gen_grader or GenerationGrader() # mistral-small (grading balance)
    
    if web_search_tool is None:
        try:
            from langchain_tavily import TavilySearch
            web_search_tool = TavilySearch(max_results=3)
        except ImportError:
            web_search_tool = None
    
    # Mistral client for generation (closed over)
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not found in environment")
    mistral_client = Mistral(api_key=api_key)
    
    # ============== Node Definitions (Closures) ==============
    
    def process_query(state: GraphState) -> GraphState:
        """Rewrite the question (using history) and route in one LLM call."""
        print("---PROCESSING QUERY (rewrite + route)---")
        question = state["question"]
        history = state.get("history", [])
        
        result = query_processor.process(question, history)
        
        print(f"  Original: {question}")
        print(f"  Rewritten: {result.query}")
        print(f"  Route: {result.route}")
        print(f"  Reasoning: {result.reasoning}")
        
        return {
            **state,
            "rewritten_query": result.query,
            "route_decision": result.route,
        }
    
    def retrieve(state: GraphState) -> GraphState:
        """Retrieve documents using hybrid search."""
        print("---RETRIEVING FROM VECTORSTORE---")
        # Use rewritten query for retrieval
        question = state.get("rewritten_query") or state["question"]
        
        # Retrieve more docs than needed - reranker will filter
        documents = retriever.retrieve(question, top_k=retrieval_top_k)
        
        print(f"  Retrieved {len(documents)} documents (will rerank to {rerank_top_k})")
        
        return {
            **state,
            "documents": documents
        }
    
    def rerank_documents(state: GraphState) -> GraphState:
        """Rerank retrieved documents by relevance to the query."""
        print("---RERANKING DOCUMENTS---")
        question = state.get("rewritten_query") or state["question"]
        documents = state["documents"]
        
        if not documents:
            print("  No documents to rerank")
            return state
        
        # Rerank and keep top_k
        reranked = doc_reranker.rerank(documents, question, top_k=rerank_top_k)
        
        print(f"  Reranked {len(documents)} → {len(reranked)} documents")
        for i, doc in enumerate(reranked[:5]):  # Show top 5
            preview = doc.content[:60].replace('\n', ' ')
            score = f"{doc.score:.3f}" if hasattr(doc, 'score') and doc.score else "N/A"
            print(f"  [{i+1}] (score={score}) {preview}...")
        
        return {
            **state,
            "documents": reranked
        }
    
    def grade_documents(state: GraphState) -> GraphState:
        """Filter retrieved documents by relevance (batch mode)."""
        print("---GRADING DOCUMENTS---")
        question = state["question"]
        documents = state["documents"]
        
        filtered_docs, needs_web_search = doc_grader.grade_documents(documents, question)
        
        print(f"  Kept {len(filtered_docs)}/{len(documents)} relevant documents")
        
        web_search_flag = "yes" if needs_web_search else "no"
        print(f"  Web search needed: {web_search_flag}")
        
        return {
            **state,
            "documents": filtered_docs,
            "web_search": web_search_flag
        }
    
    def web_search(state: GraphState) -> GraphState:
        """Perform web search as fallback. Sets web_search_done=True."""
        print("---PERFORMING WEB SEARCH---")
        question = state["question"]
        documents = list(state.get("documents", []))
        
        if web_search_tool is None:
            print("  Web search tool not configured, skipping")
            return {
                **state,
                "documents": documents,
                "web_search": "no",
                "web_search_done": True
            }
        
        try:
            search_results = web_search_tool.invoke({"query": question})
            
            # Handle different return formats from Tavily
            # TavilySearch returns a dict with a "results" key
            if isinstance(search_results, dict):
                count = 0
                for result in search_results.get("results", []):
                    content = result.get("content", "") or result.get("snippet", "")
                    url = result.get("url", "")
                    title = result.get("title", "")
                    if content:
                        doc = Document(
                            content=content,
                            metadata={"url": url, "title": title, "source": "web"},
                            source="web"
                        )
                        documents.append(doc)
                        count += 1
                print(f"  Added {count} web results")
            elif isinstance(search_results, str):
                doc = Document(
                    content=search_results,
                    metadata={"source": "web"},
                    source="web"
                )
                documents.append(doc)
                print(f"  Added 1 web result (string)")
            elif isinstance(search_results, list):
                count = 0
                for result in search_results:
                    if isinstance(result, dict):
                        content = result.get("content", "") or result.get("snippet", "") or str(result)
                        url = result.get("url", "")
                    else:
                        content = str(result)
                        url = ""
                    if content:
                        doc = Document(
                            content=content,
                            metadata={"url": url, "source": "web"} if url else {"source": "web"},
                            source="web"
                        )
                        documents.append(doc)
                        count += 1
                print(f"  Added {count} web results")
            
        except Exception as e:
            print(f"  Web search error: {e}")
        
        return {
            **state,
            "documents": documents,
            "web_search": "no",
            "web_search_done": True  # Mark that we've done web search
        }
    
    def generate(state: GraphState) -> GraphState:
        """Generate answer from documents and question."""
        print("---GENERATING ANSWER---")
        question = state["question"]
        documents = state["documents"]
        attempts = state.get("generation_attempts", 0)
        
        # Format documents as context
        context_parts = []
        for i, doc in enumerate(documents):
            source = doc.source if hasattr(doc, 'source') else "vectorstore"
            content = doc.content if hasattr(doc, 'content') else str(doc)
            context_parts.append(f"[Document {i+1}] ({source})\n{content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        system_prompt = """You are a helpful assistant answering questions based on the provided documents.

Instructions:
- Answer the question based ONLY on the provided documents
- If the documents don't contain enough information, say so
- Be concise and accurate
- Cite relevant document numbers when possible (e.g., "According to [Document 1]...")"""

        user_prompt = f"""## Context Documents:
{context}

## Question:
{question}

## Answer:"""

        response = mistral_client.chat.complete(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=2000
        )
        
        generation = response.choices[0].message.content.strip()
        
        print(f"  Generated answer ({len(generation)} chars)")
        
        return {
            **state,
            "generation": generation,
            "generation_attempts": attempts + 1
        }
    
    def grade_generation(state: GraphState) -> GraphState:
        """Grade the generation for hallucination and usefulness (single LLM call)."""
        print("---GRADING GENERATION---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        
        # Format documents for grading
        doc_texts = []
        for doc in documents:
            content = doc.content if hasattr(doc, 'content') else str(doc)
            doc_texts.append(content)
        documents_str = "\n\n---\n\n".join(doc_texts)
        
        result = gen_grader.grade(documents_str, generation, question)
        
        print(f"  Grade result: {result}")
        
        return {
            **state,
            "generation_grade": result
        }
    
    # ============== Conversational Generation ==============
    
    def generate_conversational(state: GraphState) -> GraphState:
        """
        Generate a friendly response for non-research queries.
        
        Uses ministral-3b for fast, simple conversational responses.
        Skips retrieval, grading, and citation extraction.
        """
        print("---GENERATING CONVERSATIONAL RESPONSE---")
        question = state.get("rewritten_query") or state["question"]
        
        # Use ministral-3b for fast conversational responses
        api_key = os.getenv("MISTRAL_API_KEY")
        client = Mistral(api_key=api_key)
        
        response = client.chat.complete(
            model=CONVERSATIONAL_MODEL,
            messages=[
                {"role": "system", "content": CONVERSATIONAL_SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=500,
        )
        
        generation = response.choices[0].message.content
        print(f"  Generated: {generation[:100]}...")
        
        return {
            **state,
            "generation": generation,
            "documents": [],  # No documents for conversational
            "generation_grade": "useful",  # Skip grading
        }
    
    # ============== Pure Routing Functions (no LLM calls) ==============
    
    def route_after_question_routing(state: GraphState) -> str:
        """Determine next node after route_question."""
        route = state.get("route_decision", "vectorstore")
        return route
    
    def route_after_grading(state: GraphState) -> str:
        """Determine next node after grade_documents."""
        web_search_needed = state.get("web_search", "no")
        
        if web_search_needed == "yes":
            return "web_search"
        return "generate"
    
    def route_after_grade_generation(state: GraphState) -> str:
        """
        Determine next node after grade_generation.
        
        Pure function - reads state, no LLM calls.
        
        Returns:
            "useful" -> END
            "retry" -> web_search (only if web_search_done is False)
            "end_degraded" -> END (accept current answer despite grade)
        """
        grade = state.get("generation_grade", "useful")
        attempts = state.get("generation_attempts", 0)
        web_done = state.get("web_search_done", False)
        
        if grade == "useful":
            return "useful"
        
        # If grade is not useful, decide whether to retry
        if attempts >= 3:
            print("  Max attempts reached, accepting current answer")
            return "end_degraded"
        
        if web_done:
            print("  Web search already done, accepting current answer")
            return "end_degraded"
        
        # Retry via web search
        return "retry"
    
    # ============== Build Graph ==============
    
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("process_query", process_query)
    workflow.add_node("generate_conversational", generate_conversational)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rerank_documents", rerank_documents)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)
    workflow.add_node("grade_generation", grade_generation)
    
    # Set entry point - start with combined rewrite + route
    workflow.set_entry_point("process_query")
    
    # Conditional Edge: process_query → conversational, vectorstore, OR web_search
    workflow.add_conditional_edges(
        "process_query",
        route_after_question_routing,
        {
            "conversational": "generate_conversational",
            "vectorstore": "retrieve",
            "web_search": "web_search",
        }
    )
    
    # Edge: generate_conversational → END (skip grading for conversational)
    workflow.add_edge("generate_conversational", END)
    
    # Edge: retrieve → rerank_documents → grade_documents
    workflow.add_edge("retrieve", "rerank_documents")
    workflow.add_edge("rerank_documents", "grade_documents")
    
    # Conditional Edge: grade_documents → web_search OR generate
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "web_search": "web_search",
            "generate": "generate",
        }
    )
    
    # Edge: web_search → generate
    workflow.add_edge("web_search", "generate")
    
    # Edge: generate → grade_generation
    workflow.add_edge("generate", "grade_generation")
    
    # Conditional Edge: grade_generation → END or retry
    workflow.add_conditional_edges(
        "grade_generation",
        route_after_grade_generation,
        {
            "useful": END,
            "end_degraded": END,
            "retry": "web_search",
        }
    )
    
    return workflow


def compile_crag_graph(
    retriever: RetrieverType,
    query_processor: Optional[QueryRewriterAndRouter] = None,
    doc_grader: Optional[DocumentGrader] = None,
    doc_reranker: Optional[DocumentReranker] = None,
    gen_grader: Optional[GenerationGrader] = None,
    web_search_tool = None,
    retrieval_top_k: int = 50,
    rerank_top_k: int = 10,
):
    """
    Create and compile the CRAG workflow.
    
    Args:
        retriever: Required HybridRetriever for vectorstore
        query_processor: Optional QueryRewriterAndRouter for rewrite + route
        doc_grader: Optional DocumentGrader
        doc_reranker: Optional DocumentReranker
        gen_grader: Optional GenerationGrader
        web_search_tool: Optional web search tool
        retrieval_top_k: Number of docs to retrieve before reranking
        rerank_top_k: Number of docs to keep after reranking
        
    Returns:
        Compiled graph ready for execution
    """
    workflow = create_crag_graph(
        retriever=retriever,
        query_processor=query_processor,
        doc_grader=doc_grader,
        doc_reranker=doc_reranker,
        gen_grader=gen_grader,
        web_search_tool=web_search_tool,
        retrieval_top_k=retrieval_top_k,
        rerank_top_k=rerank_top_k,
    )
    return workflow.compile()


def run_crag(
    question: str,
    retriever: RetrieverType,
    history: Optional[List[Message]] = None,
    query_processor: Optional[QueryRewriterAndRouter] = None,
    doc_grader: Optional[DocumentGrader] = None,
    doc_reranker: Optional[DocumentReranker] = None,
    gen_grader: Optional[GenerationGrader] = None,
    web_search_tool = None,
    documents: Optional[List[Document]] = None,
    retrieval_top_k: int = 50,
    rerank_top_k: int = 10,
) -> dict:
    """
    Run the CRAG workflow on a question.
    
    Args:
        question: The user's question
        retriever: Required Retriever instance (PineconeHybridRetriever recommended)
        history: Optional conversation history for multi-turn support
        query_processor: Optional QueryRewriterAndRouter for rewrite + route
        doc_grader: Optional DocumentGrader
        doc_reranker: Optional DocumentReranker
        gen_grader: Optional GenerationGrader
        web_search_tool: Optional web search tool
        documents: Optional pre-loaded documents (for testing)
        retrieval_top_k: Number of docs to retrieve before reranking (default: 50)
        rerank_top_k: Number of docs to keep after reranking (default: 10)
        
    Returns:
        Final state dictionary with:
        - question: Original question
        - rewritten_query: Standalone query (after history rewriting)
        - history: Conversation history
        - documents: Retrieved/filtered documents
        - generation: Final answer
        - web_search: Whether web search flag was set
        - web_search_done: Whether web search was performed
        - generation_attempts: Number of generation attempts
        - generation_grade: Result of grading
        - route_decision: Initial routing decision
        
    Example:
        >>> from crag import run_crag, PineconeHybridRetriever, Document
        >>> 
        >>> # Create Pinecone retriever (persistent)
        >>> retriever = PineconeHybridRetriever(index_name="my-index")
        >>> 
        >>> # Add documents (one-time indexing)
        >>> docs = [
        ...     Document(content="GPT-4 achieved 86.4% on MMLU benchmark."),
        ...     Document(content="Claude 2 scored 78.5% on MMLU."),
        ... ]
        >>> retriever.add_documents(docs)
        >>> 
        >>> # Run CRAG (uses persistent Pinecone index)
        >>> result = run_crag("What is GPT-4's MMLU score?", retriever=retriever)
        >>> print(result["generation"])
        >>> 
        >>> # Multi-turn with history
        >>> history = [
        ...     {"role": "user", "content": "Tell me about GPT-4"},
        ...     {"role": "assistant", "content": "GPT-4 is a large multimodal model..."},
        ... ]
        >>> result = run_crag("What about its MMLU score?", retriever=retriever, history=history)
    """
    # Initialize state with all fields
    initial_state: GraphState = {
        "question": question,
        "rewritten_query": None,
        "history": history or [],
        "documents": documents or [],
        "generation": "",
        "web_search": "no",
        "web_search_done": False,
        "generation_attempts": 0,
        "generation_grade": None,
        "route_decision": None,
    }
    
    # Compile and run graph
    app = compile_crag_graph(
        retriever=retriever,
        query_processor=query_processor,
        doc_grader=doc_grader,
        doc_reranker=doc_reranker,
        gen_grader=gen_grader,
        web_search_tool=web_search_tool,
        retrieval_top_k=retrieval_top_k,
        rerank_top_k=rerank_top_k,
    )
    
    print("=" * 60)
    print(f"CRAG Query: {question}")
    if history:
        print(f"  (with {len(history)} history messages)")
    print("=" * 60)
    
    # Execute the graph
    final_state = app.invoke(initial_state)
    
    print("=" * 60)
    print("CRAG Complete")
    if final_state.get('rewritten_query') != question:
        print(f"  Rewritten: {final_state.get('rewritten_query')}")
    print(f"  Route: {final_state.get('route_decision')}")
    print(f"  Documents: {len(final_state.get('documents', []))}")
    print(f"  Attempts: {final_state.get('generation_attempts')}")
    print(f"  Grade: {final_state.get('generation_grade')}")
    print(f"  Web search done: {final_state.get('web_search_done')}")
    print("=" * 60)
    
    return final_state


def stream_crag(
    question: str,
    retriever: RetrieverType,
    history: Optional[List[Message]] = None,
    query_processor: Optional[QueryRewriterAndRouter] = None,
    doc_grader: Optional[DocumentGrader] = None,
    doc_reranker: Optional[DocumentReranker] = None,
    gen_grader: Optional[GenerationGrader] = None,
    web_search_tool = None,
    retrieval_top_k: int = 50,
    rerank_top_k: int = 10,
):
    """
    Stream the CRAG workflow execution step by step.
    
    Yields state after each node execution.
    
    Args:
        question: The user's question
        retriever: Required HybridRetriever instance
        history: Optional conversation history for multi-turn support
        query_processor: Optional QueryRewriterAndRouter for rewrite + route
        doc_grader: Optional DocumentGrader
        doc_reranker: Optional DocumentReranker
        gen_grader: Optional GenerationGrader
        web_search_tool: Optional web search tool
        retrieval_top_k: Number of docs to retrieve before reranking
        rerank_top_k: Number of docs to keep after reranking
        
    Yields:
        Tuple of (node_name, state) after each step
        
    Example:
        >>> for node, state in stream_crag("What is MMLU?", retriever=retriever):
        ...     print(f"After {node}: {len(state.get('documents', []))} docs")
    """
    initial_state: GraphState = {
        "question": question,
        "rewritten_query": None,
        "history": history or [],
        "documents": [],
        "generation": "",
        "web_search": "no",
        "web_search_done": False,
        "generation_attempts": 0,
        "generation_grade": None,
        "route_decision": None,
    }
    
    app = compile_crag_graph(
        retriever=retriever,
        query_processor=query_processor,
        doc_grader=doc_grader,
        doc_reranker=doc_reranker,
        gen_grader=gen_grader,
        web_search_tool=web_search_tool,
        retrieval_top_k=retrieval_top_k,
        rerank_top_k=rerank_top_k,
    )
    
    for output in app.stream(initial_state):
        for node_name, state in output.items():
            yield node_name, state
