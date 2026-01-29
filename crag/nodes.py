"""
CRAG Workflow Nodes.

Each node is a function that takes GraphState and returns updated state.
Nodes perform the core logic: routing, retrieval, grading, generation.

All LLM operations use Mistral Large.
"""

import os
from typing import List, Any, Union
from langchain_community.tools.tavily_search import TavilySearchResults
from mistralai import Mistral

from .graph_state import GraphState
from .retrieval import Document, HybridRetriever, PineconeHybridRetriever, InMemoryHybridRetriever
from .graders import QueryRouter, DocumentGrader, GenerationGrader, DEFAULT_MODEL


# ============== Global Components (initialized lazily) ==============

_retriever: Union[HybridRetriever, PineconeHybridRetriever, InMemoryHybridRetriever, None] = None
_router: QueryRouter = None
_doc_grader: DocumentGrader = None
_gen_grader: GenerationGrader = None
_mistral_client: Mistral = None
_web_search_tool: TavilySearchResults = None


def set_retriever(retriever: Union[HybridRetriever, PineconeHybridRetriever, InMemoryHybridRetriever]):
    """Set the hybrid retriever for the workflow."""
    global _retriever
    _retriever = retriever


def get_retriever() -> Union[HybridRetriever, PineconeHybridRetriever, InMemoryHybridRetriever]:
    """Get the hybrid retriever."""
    if _retriever is None:
        raise ValueError("Retriever not initialized. Call set_retriever() first.")
    return _retriever


def get_router() -> QueryRouter:
    """Get or create the query router (uses Mistral Large)."""
    global _router
    if _router is None:
        _router = QueryRouter(model=DEFAULT_MODEL)
    return _router


def get_doc_grader() -> DocumentGrader:
    """Get or create the document grader (uses Mistral Large)."""
    global _doc_grader
    if _doc_grader is None:
        _doc_grader = DocumentGrader(model=DEFAULT_MODEL)
    return _doc_grader


def get_gen_grader() -> GenerationGrader:
    """Get or create the generation grader (uses Mistral Large)."""
    global _gen_grader
    if _gen_grader is None:
        _gen_grader = GenerationGrader(model=DEFAULT_MODEL)
    return _gen_grader


def get_mistral_client() -> Mistral:
    """Get or create Mistral client for generation."""
    global _mistral_client
    if _mistral_client is None:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment")
        _mistral_client = Mistral(api_key=api_key)
    return _mistral_client


def get_web_search_tool() -> TavilySearchResults:
    """Get or create the web search tool."""
    global _web_search_tool
    if _web_search_tool is None:
        _web_search_tool = TavilySearchResults(max_results=3)
    return _web_search_tool


# ============== Node A: Route Question ==============

def route_question(state: GraphState) -> GraphState:
    """
    Route the question to vectorstore or web search.
    
    Uses an LLM to classify the query:
    - Technical papers/indexed content → vectorstore
    - General knowledge/current events → web_search
    """
    print("---ROUTING QUESTION---")
    question = state["question"]
    
    router = get_router()
    result = router.route(question)
    
    print(f"  Route decision: {result.datasource}")
    print(f"  Reasoning: {result.reasoning}")
    
    return {
        **state,
        "route_decision": result.datasource
    }


# ============== Node B: Retrieve from Vector Store ==============

def retrieve(state: GraphState) -> GraphState:
    """
    Retrieve documents using hybrid search (BM25 + Semantic + RRF).
    
    Performs:
    1. BM25 keyword search for exact matches
    2. Semantic embedding search for meaning
    3. RRF fusion to combine results
    """
    print("---RETRIEVING FROM VECTORSTORE---")
    question = state["question"]
    
    retriever = get_retriever()
    documents = retriever.retrieve(question, top_k=5)
    
    print(f"  Retrieved {len(documents)} documents")
    for i, doc in enumerate(documents):
        preview = doc.content[:80].replace('\n', ' ')
        print(f"  [{i+1}] {preview}...")
    
    return {
        **state,
        "documents": documents
    }


# ============== Node C: Grade Documents ==============

def grade_documents(state: GraphState) -> GraphState:
    """
    Filter retrieved documents by relevance.
    
    Uses an LLM grader to check each document:
    - Relevant → keep
    - Irrelevant → filter out
    
    Sets web_search="yes" if filtered list is empty.
    """
    print("---GRADING DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]
    
    grader = get_doc_grader()
    filtered_docs, needs_web_search = grader.grade_documents(documents, question)
    
    print(f"  Kept {len(filtered_docs)}/{len(documents)} relevant documents")
    
    web_search_flag = "yes" if needs_web_search else "no"
    print(f"  Web search needed: {web_search_flag}")
    
    return {
        **state,
        "documents": filtered_docs,
        "web_search": web_search_flag
    }


# ============== Node D: Web Search ==============

def web_search(state: GraphState) -> GraphState:
    """
    Perform web search as fallback.
    
    Fetches top search results and converts them to Document format.
    Appends findings to existing documents.
    """
    print("---PERFORMING WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])
    
    tool = get_web_search_tool()
    
    try:
        search_results = tool.invoke({"query": question})
        
        # Convert search results to Documents
        for result in search_results:
            content = result.get("content", "")
            url = result.get("url", "")
            
            doc = Document(
                content=content,
                metadata={"url": url, "source": "web"},
                source="web"
            )
            documents.append(doc)
        
        print(f"  Added {len(search_results)} web results")
        
    except Exception as e:
        print(f"  Web search error: {e}")
    
    return {
        **state,
        "documents": documents,
        "web_search": "no"  # Reset flag after searching
    }


# ============== Node E: Generate ==============

def generate(state: GraphState) -> GraphState:
    """
    Generate answer from documents and question.
    
    Passes the question and consolidated documents to Mistral Large
    to synthesize an answer.
    """
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
    
    # Generate with Mistral Large
    client = get_mistral_client()
    
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

    response = client.chat.complete(
        model=DEFAULT_MODEL,
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


# ============== Node F: Grade Generation ==============

def grade_generation(state: GraphState) -> str:
    """
    Perform quality check on generation.
    
    Two-step verification:
    1. Hallucination Check: Is generation grounded in documents?
    2. Answer Check: Does generation answer the question?
    
    Returns:
        - "useful": Proceed to END
        - "not supported": Hallucination detected, retry
        - "not useful": Doesn't answer question, retry
    """
    print("---GRADING GENERATION---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    attempts = state.get("generation_attempts", 0)
    
    # Format documents for grading
    doc_texts = []
    for doc in documents:
        content = doc.content if hasattr(doc, 'content') else str(doc)
        doc_texts.append(content)
    documents_str = "\n\n---\n\n".join(doc_texts)
    
    grader = get_gen_grader()
    result = grader.grade(documents_str, generation, question)
    
    print(f"  Grade result: {result}")
    print(f"  Generation attempts: {attempts}")
    
    # Prevent infinite loops - max 3 attempts
    if attempts >= 3 and result != "useful":
        print("  Max attempts reached, returning current answer")
        return "useful"
    
    return result


# ============== Routing Functions for Conditional Edges ==============

def route_after_question_routing(state: GraphState) -> str:
    """
    Determine next node after route_question.
    
    Returns:
        "vectorstore" or "web_search"
    """
    route = state.get("route_decision", "vectorstore")
    return route


def route_after_grading(state: GraphState) -> str:
    """
    Determine next node after grade_documents.
    
    Returns:
        "web_search" if documents were irrelevant
        "generate" if documents are good
    """
    web_search = state.get("web_search", "no")
    
    if web_search == "yes":
        return "web_search"
    return "generate"


def route_after_generation(state: GraphState) -> str:
    """
    Determine next node after generation grading.
    
    Returns:
        "useful" → END
        "not supported" or "not useful" → retry via web_search
    """
    return grade_generation(state)
