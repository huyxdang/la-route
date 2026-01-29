"""
CRAG LangGraph Workflow.

Builds the stateful graph with nodes and conditional edges.
All LLM operations use Mistral Large.
Vector storage uses Pinecone with hybrid search.
"""

from typing import List, Optional, Union
from langgraph.graph import StateGraph, END

from .graph_state import GraphState
from .retrieval import Document, HybridRetriever, PineconeHybridRetriever, InMemoryHybridRetriever
from .nodes import (
    route_question,
    retrieve,
    grade_documents,
    web_search,
    generate,
    route_after_question_routing,
    route_after_grading,
    route_after_generation,
    set_retriever,
)

# Type alias for any retriever type
RetrieverType = Union[HybridRetriever, PineconeHybridRetriever, InMemoryHybridRetriever]


def create_crag_graph(retriever: Optional[RetrieverType] = None) -> StateGraph:
    """
    Create the CRAG workflow graph.
    
    Graph Structure:
    ```
    START → route_question
                ↓
        ┌───────┴───────┐
        ↓               ↓
    vectorstore    web_search
        ↓               │
    retrieve            │
        ↓               │
    grade_documents     │
        ↓               │
        ├───────────────┤
        ↓               ↓
    (if irrelevant)  generate ←─┐
        ↓               ↓       │
    web_search     grade_gen    │
        ↓               ↓       │
        └──→ generate   ├───────┘ (if not useful)
                        ↓
                       END (if useful)
    ```
    
    Args:
        retriever: Optional HybridRetriever instance. If provided, will be 
                   used for vectorstore retrieval.
                   
    Returns:
        Compiled LangGraph workflow
    """
    # Set retriever if provided
    if retriever is not None:
        set_retriever(retriever)
    
    # Initialize the graph with our state schema
    workflow = StateGraph(GraphState)
    
    # ============== Add Nodes ==============
    
    # Node A: Route the question
    workflow.add_node("route_question", route_question)
    
    # Node B: Retrieve from vectorstore  
    workflow.add_node("retrieve", retrieve)
    
    # Node C: Grade documents
    workflow.add_node("grade_documents", grade_documents)
    
    # Node D: Web search fallback
    workflow.add_node("web_search", web_search)
    
    # Node E: Generate answer
    workflow.add_node("generate", generate)
    
    # ============== Add Edges ==============
    
    # Start → route_question
    workflow.set_entry_point("route_question")
    
    # Conditional Edge: route_question → vectorstore OR web_search
    workflow.add_conditional_edges(
        "route_question",
        route_after_question_routing,
        {
            "vectorstore": "retrieve",
            "web_search": "web_search",
        }
    )
    
    # Edge: retrieve → grade_documents
    workflow.add_edge("retrieve", "grade_documents")
    
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
    
    # Conditional Edge: generate → END or retry
    workflow.add_conditional_edges(
        "generate",
        route_after_generation,
        {
            "useful": END,
            "not useful": "web_search",
            "not supported": "web_search",
        }
    )
    
    return workflow


def compile_crag_graph(retriever: Optional[RetrieverType] = None):
    """
    Create and compile the CRAG workflow.
    
    Args:
        retriever: Optional HybridRetriever for vectorstore
        
    Returns:
        Compiled graph ready for execution
    """
    workflow = create_crag_graph(retriever)
    return workflow.compile()


def run_crag(
    question: str,
    retriever: Optional[RetrieverType] = None,
    documents: Optional[List[Document]] = None,
) -> dict:
    """
    Run the CRAG workflow on a question.
    
    Args:
        question: The user's question
        retriever: Retriever instance (PineconeHybridRetriever recommended for production)
        documents: Optional pre-loaded documents (for testing)
        
    Returns:
        Final state dictionary with:
        - question: Original question
        - documents: Retrieved/filtered documents
        - generation: Final answer
        - web_search: Whether web search was used
        - generation_attempts: Number of generation attempts
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
    """
    # Initialize state
    initial_state: GraphState = {
        "question": question,
        "documents": documents or [],
        "generation": "",
        "web_search": "no",
        "generation_attempts": 0,
        "route_decision": None,
    }
    
    # Compile and run graph
    app = compile_crag_graph(retriever)
    
    print("=" * 60)
    print(f"CRAG Query: {question}")
    print("=" * 60)
    
    # Execute the graph
    final_state = app.invoke(initial_state)
    
    print("=" * 60)
    print("CRAG Complete")
    print(f"  Route: {final_state.get('route_decision')}")
    print(f"  Documents: {len(final_state.get('documents', []))}")
    print(f"  Attempts: {final_state.get('generation_attempts')}")
    print("=" * 60)
    
    return final_state


def stream_crag(
    question: str,
    retriever: Optional[RetrieverType] = None,
):
    """
    Stream the CRAG workflow execution step by step.
    
    Yields state after each node execution.
    
    Args:
        question: The user's question
        retriever: HybridRetriever instance
        
    Yields:
        Tuple of (node_name, state) after each step
        
    Example:
        >>> for node, state in stream_crag("What is MMLU?", retriever):
        ...     print(f"After {node}: {len(state.get('documents', []))} docs")
    """
    initial_state: GraphState = {
        "question": question,
        "documents": [],
        "generation": "",
        "web_search": "no",
        "generation_attempts": 0,
        "route_decision": None,
    }
    
    app = compile_crag_graph(retriever)
    
    for output in app.stream(initial_state):
        for node_name, state in output.items():
            yield node_name, state
