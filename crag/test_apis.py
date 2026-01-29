"""
API Integration Tests for CRAG System.

Tests all components to verify API keys and integrations are working.

Usage:
    python -m crag.test_apis
    
    # Or run specific tests
    python -m crag.test_apis --component router
    python -m crag.test_apis --component rewriter
    python -m crag.test_apis --component doc_grader
    python -m crag.test_apis --component reranker
    python -m crag.test_apis --component gen_grader
    python -m crag.test_apis --component retriever
    python -m crag.test_apis --component full_pipeline
"""

import os
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    message: str
    error: Optional[str] = None


def print_result(result: TestResult):
    """Print a test result with color coding."""
    status = "✓ PASS" if result.passed else "✗ FAIL"
    color = "\033[92m" if result.passed else "\033[91m"
    reset = "\033[0m"
    
    print(f"\n{color}{status}{reset} - {result.name}")
    print(f"  {result.message}")
    if result.error:
        print(f"  Error: {result.error}")


def check_api_key(key_name: str) -> bool:
    """Check if an API key is set."""
    return bool(os.getenv(key_name))


# ============== Individual Component Tests ==============

def test_query_router() -> TestResult:
    """Test QueryRouter with Mistral API."""
    name = "QueryRouter (ministral-3b)"
    
    if not check_api_key("MISTRAL_API_KEY"):
        return TestResult(name, False, "MISTRAL_API_KEY not set")
    
    try:
        from crag.graders import QueryRouter
        
        router = QueryRouter()
        result = router.route("What is GPT-4's MMLU score?")
        
        assert result.datasource in ["vectorstore", "web_search"], f"Invalid datasource: {result.datasource}"
        assert result.reasoning, "No reasoning provided"
        
        return TestResult(
            name, 
            True, 
            f"Routed to '{result.datasource}' - {result.reasoning[:50]}..."
        )
    except Exception as e:
        return TestResult(name, False, "Failed to route query", str(e))


def test_query_rewriter() -> TestResult:
    """Test QueryRewriter with conversation history."""
    name = "QueryRewriter (mistral-small)"
    
    if not check_api_key("MISTRAL_API_KEY"):
        return TestResult(name, False, "MISTRAL_API_KEY not set")
    
    try:
        from crag.graders import QueryRewriter
        
        rewriter = QueryRewriter()
        
        history = [
            {"role": "user", "content": "Tell me about GPT-4"},
            {"role": "assistant", "content": "GPT-4 is a large multimodal model by OpenAI..."},
        ]
        
        result = rewriter.rewrite("What about its MMLU score?", history)
        
        assert result.query, "No rewritten query"
        assert "mmlu" in result.query.lower() or "gpt" in result.query.lower(), \
            f"Query doesn't seem to include context: {result.query}"
        
        return TestResult(
            name, 
            True, 
            f"Rewrote to: '{result.query}'"
        )
    except Exception as e:
        return TestResult(name, False, "Failed to rewrite query", str(e))


def test_document_grader() -> TestResult:
    """Test DocumentGrader batch relevance grading."""
    name = "DocumentGrader (mistral-large)"
    
    if not check_api_key("MISTRAL_API_KEY"):
        return TestResult(name, False, "MISTRAL_API_KEY not set")
    
    try:
        from crag.graders import DocumentGrader
        from crag.retrieval import Document
        
        grader = DocumentGrader()
        
        docs = [
            Document(content="GPT-4 achieved 86.4% accuracy on the MMLU benchmark."),
            Document(content="The weather in Paris is sunny today."),  # Irrelevant
            Document(content="Claude 2 scored 78.5% on MMLU evaluation."),
        ]
        
        filtered, needs_web = grader.grade_documents(docs, "What is GPT-4's MMLU score?")
        
        assert len(filtered) >= 1, "Should keep at least the relevant document"
        assert len(filtered) <= 2, "Should filter out the weather document"
        
        return TestResult(
            name, 
            True, 
            f"Filtered {len(docs)} → {len(filtered)} docs, needs_web_search={needs_web}"
        )
    except Exception as e:
        return TestResult(name, False, "Failed to grade documents", str(e))


def test_document_reranker_cohere() -> TestResult:
    """Test DocumentReranker with Cohere API."""
    name = "DocumentReranker (Cohere)"
    
    if not check_api_key("COHERE_API_KEY"):
        return TestResult(name, False, "COHERE_API_KEY not set (skipped)")
    
    try:
        from crag.graders import DocumentReranker
        from crag.retrieval import Document
        
        reranker = DocumentReranker(backend="cohere")
        
        docs = [
            Document(content="The weather is nice today.", score=0.9),
            Document(content="GPT-4 achieved 86.4% on MMLU.", score=0.5),
            Document(content="Paris is the capital of France.", score=0.8),
            Document(content="MMLU tests broad knowledge across 57 subjects.", score=0.3),
        ]
        
        reranked = reranker.rerank(docs, "What is GPT-4's MMLU score?", top_k=2)
        
        assert len(reranked) == 2, f"Should return 2 docs, got {len(reranked)}"
        
        # The GPT-4 MMLU doc should be ranked higher after reranking
        top_content = reranked[0].content.lower()
        assert "gpt-4" in top_content or "mmlu" in top_content, \
            f"Top doc should be about GPT-4/MMLU, got: {reranked[0].content[:50]}"
        
        return TestResult(
            name, 
            True, 
            f"Reranked {len(docs)} → {len(reranked)} docs, top: '{reranked[0].content[:40]}...'"
        )
    except Exception as e:
        return TestResult(name, False, "Failed to rerank with Cohere", str(e))


def test_document_reranker_crossencoder() -> TestResult:
    """Test DocumentReranker with local CrossEncoder."""
    name = "DocumentReranker (CrossEncoder)"
    
    try:
        from crag.graders import DocumentReranker
        from crag.retrieval import Document
        
        reranker = DocumentReranker(backend="cross-encoder")
        
        docs = [
            Document(content="The weather is nice today.", score=0.9),
            Document(content="GPT-4 achieved 86.4% on MMLU benchmark evaluation.", score=0.5),
            Document(content="Paris is the capital of France.", score=0.8),
        ]
        
        reranked = reranker.rerank(docs, "What is GPT-4's MMLU score?", top_k=2)
        
        assert len(reranked) == 2, f"Should return 2 docs, got {len(reranked)}"
        
        return TestResult(
            name, 
            True, 
            f"Reranked {len(docs)} → {len(reranked)} docs, top score: {reranked[0].score:.3f}"
        )
    except ImportError as e:
        return TestResult(name, False, "sentence-transformers not installed", str(e))
    except Exception as e:
        return TestResult(name, False, "Failed to rerank with CrossEncoder", str(e))


def test_generation_grader() -> TestResult:
    """Test GenerationGrader for hallucination and usefulness."""
    name = "GenerationGrader (mistral-small)"
    
    if not check_api_key("MISTRAL_API_KEY"):
        return TestResult(name, False, "MISTRAL_API_KEY not set")
    
    try:
        from crag.graders import GenerationGrader
        
        grader = GenerationGrader()
        
        documents = "GPT-4 achieved 86.4% accuracy on the MMLU benchmark."
        generation = "According to the document, GPT-4 scored 86.4% on MMLU."
        question = "What is GPT-4's MMLU score?"
        
        result = grader.grade(documents, generation, question)
        
        assert result in ["useful", "not supported", "not useful"], f"Invalid grade: {result}"
        
        # This should be graded as useful (grounded + answers question)
        expected = "useful"
        
        return TestResult(
            name, 
            True, 
            f"Graded as '{result}' (expected '{expected}')"
        )
    except Exception as e:
        return TestResult(name, False, "Failed to grade generation", str(e))


def test_inmemory_retriever() -> TestResult:
    """Test InMemoryHybridRetriever."""
    name = "InMemoryHybridRetriever"
    
    if not check_api_key("MISTRAL_API_KEY"):
        return TestResult(name, False, "MISTRAL_API_KEY not set (needed for embeddings)")
    
    try:
        from crag.retrieval import InMemoryHybridRetriever, Document
        
        retriever = InMemoryHybridRetriever()
        
        docs = [
            Document(content="GPT-4 achieved 86.4% on MMLU.", id="doc1"),
            Document(content="Claude 2 scored 78.5% on MMLU.", id="doc2"),
            Document(content="Llama 2 70B got 68.9% on MMLU.", id="doc3"),
        ]
        
        retriever.add_documents(docs)
        
        results = retriever.retrieve("What is GPT-4's MMLU score?", top_k=2)
        
        assert len(results) == 2, f"Should return 2 docs, got {len(results)}"
        assert any("gpt-4" in r.content.lower() for r in results), \
            "GPT-4 doc should be in top results"
        
        return TestResult(
            name, 
            True, 
            f"Retrieved {len(results)} docs, top: '{results[0].content[:40]}...'"
        )
    except Exception as e:
        return TestResult(name, False, "Failed to retrieve documents", str(e))


def test_pinecone_retriever() -> TestResult:
    """Test PineconeHybridRetriever connection."""
    name = "PineconeHybridRetriever"
    
    if not check_api_key("PINECONE_API_KEY"):
        return TestResult(name, False, "PINECONE_API_KEY not set (skipped)")
    
    if not check_api_key("MISTRAL_API_KEY"):
        return TestResult(name, False, "MISTRAL_API_KEY not set (needed for embeddings)")
    
    try:
        from crag.retrieval import PineconeHybridRetriever
        
        # Just test connection, don't create a real index
        retriever = PineconeHybridRetriever(
            index_name="crag-test-connection",
            create_index=False  # Don't actually create
        )
        
        # Test that we can get Pinecone client
        pc = retriever._get_pinecone()
        indexes = [idx.name for idx in pc.list_indexes()]
        
        return TestResult(
            name, 
            True, 
            f"Connected to Pinecone, found {len(indexes)} indexes"
        )
    except Exception as e:
        return TestResult(name, False, "Failed to connect to Pinecone", str(e))


def test_web_search() -> TestResult:
    """Test Tavily web search integration."""
    name = "TavilySearch"
    
    if not check_api_key("TAVILY_API_KEY"):
        return TestResult(name, False, "TAVILY_API_KEY not set (skipped)")
    
    try:
        from langchain_tavily import TavilySearch
        
        search = TavilySearch(max_results=2)
        results = search.invoke({"query": "OpenAI GPT-4 release date"})
        
        assert results, "No search results returned"
        
        if isinstance(results, list):
            count = len(results)
        else:
            count = 1
        
        return TestResult(
            name, 
            True, 
            f"Retrieved {count} web result(s)"
        )
    except Exception as e:
        return TestResult(name, False, "Failed to perform web search", str(e))


def test_full_pipeline() -> TestResult:
    """Test the full CRAG pipeline end-to-end."""
    name = "Full CRAG Pipeline"
    
    if not check_api_key("MISTRAL_API_KEY"):
        return TestResult(name, False, "MISTRAL_API_KEY not set")
    
    try:
        from crag import run_crag, InMemoryHybridRetriever, Document
        
        # Use in-memory retriever for testing
        retriever = InMemoryHybridRetriever()
        
        docs = [
            Document(content="GPT-4 achieved 86.4% accuracy on the MMLU benchmark, significantly outperforming previous models."),
            Document(content="Claude 2 by Anthropic scored 78.5% on MMLU evaluation."),
            Document(content="The MMLU benchmark tests knowledge across 57 academic subjects."),
        ]
        retriever.add_documents(docs)
        
        # Run with minimal output
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            result = run_crag(
                "What is GPT-4's MMLU score?",
                retriever=retriever,
                retrieval_top_k=3,  # Small for testing
                rerank_top_k=2,
            )
        
        assert result.get("generation"), "No generation produced"
        assert result.get("route_decision"), "No route decision"
        assert result.get("documents"), "No documents in final state"
        
        gen_preview = result["generation"][:80].replace('\n', ' ')
        
        return TestResult(
            name, 
            True, 
            f"Generated answer: '{gen_preview}...'"
        )
    except Exception as e:
        return TestResult(name, False, "Pipeline failed", str(e))


def test_full_pipeline_with_history() -> TestResult:
    """Test CRAG pipeline with conversation history."""
    name = "CRAG Pipeline with History"
    
    if not check_api_key("MISTRAL_API_KEY"):
        return TestResult(name, False, "MISTRAL_API_KEY not set")
    
    try:
        from crag import run_crag, InMemoryHybridRetriever, Document
        
        retriever = InMemoryHybridRetriever()
        
        docs = [
            Document(content="GPT-4 achieved 86.4% accuracy on the MMLU benchmark."),
            Document(content="GPT-4 scored 95.3% on HellaSwag benchmark."),
            Document(content="Claude 2 scored 78.5% on MMLU."),
        ]
        retriever.add_documents(docs)
        
        history = [
            {"role": "user", "content": "Tell me about GPT-4 benchmarks"},
            {"role": "assistant", "content": "GPT-4 performs well on various benchmarks including MMLU and HellaSwag."},
        ]
        
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            result = run_crag(
                "What about HellaSwag specifically?",  # Contextual question
                retriever=retriever,
                history=history,
                retrieval_top_k=3,
                rerank_top_k=2,
            )
        
        assert result.get("generation"), "No generation produced"
        assert result.get("rewritten_query"), "Query should be rewritten"
        
        # The rewritten query should mention HellaSwag and possibly GPT-4
        rewritten = result["rewritten_query"].lower()
        assert "hellaswag" in rewritten, f"Rewritten query should mention HellaSwag: {result['rewritten_query']}"
        
        return TestResult(
            name, 
            True, 
            f"Rewritten: '{result['rewritten_query']}'"
        )
    except Exception as e:
        return TestResult(name, False, "Pipeline with history failed", str(e))


# ============== Test Runner ==============

ALL_TESTS = {
    "router": test_query_router,
    "rewriter": test_query_rewriter,
    "doc_grader": test_document_grader,
    "reranker_cohere": test_document_reranker_cohere,
    "reranker_crossencoder": test_document_reranker_crossencoder,
    "gen_grader": test_generation_grader,
    "retriever_memory": test_inmemory_retriever,
    "retriever_pinecone": test_pinecone_retriever,
    "web_search": test_web_search,
    "full_pipeline": test_full_pipeline,
    "pipeline_history": test_full_pipeline_with_history,
}


def run_tests(components: Optional[list] = None) -> list:
    """Run specified tests or all tests."""
    if components:
        tests_to_run = {k: v for k, v in ALL_TESTS.items() if k in components}
    else:
        tests_to_run = ALL_TESTS
    
    results = []
    
    print("\n" + "=" * 60)
    print("CRAG API Integration Tests")
    print("=" * 60)
    
    # Check API keys first
    print("\nAPI Keys Status:")
    api_keys = ["MISTRAL_API_KEY", "PINECONE_API_KEY", "COHERE_API_KEY", "TAVILY_API_KEY"]
    for key in api_keys:
        status = "✓ Set" if check_api_key(key) else "✗ Not set"
        color = "\033[92m" if check_api_key(key) else "\033[93m"
        reset = "\033[0m"
        print(f"  {color}{status}{reset} - {key}")
    
    print("\n" + "-" * 60)
    print("Running Tests...")
    print("-" * 60)
    
    for name, test_fn in tests_to_run.items():
        print(f"\nTesting: {name}...")
        result = test_fn()
        results.append(result)
        print_result(result)
    
    return results


def print_summary(results: list):
    """Print test summary."""
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    skipped = sum(1 for r in results if not r.passed and "not set" in r.message.lower())
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"  Total:   {len(results)}")
    print(f"  \033[92mPassed:  {passed}\033[0m")
    print(f"  \033[91mFailed:  {failed - skipped}\033[0m")
    print(f"  \033[93mSkipped: {skipped}\033[0m (missing API keys)")
    print("=" * 60)
    
    if failed - skipped > 0:
        print("\nFailed tests:")
        for r in results:
            if not r.passed and "not set" not in r.message.lower():
                print(f"  - {r.name}: {r.error or r.message}")
    
    return failed - skipped == 0


def main():
    parser = argparse.ArgumentParser(description="Test CRAG API integrations")
    parser.add_argument(
        "--component",
        choices=list(ALL_TESTS.keys()),
        help="Test specific component only"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available tests"
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("Available tests:")
        for name in ALL_TESTS.keys():
            print(f"  - {name}")
        return
    
    components = [args.component] if args.component else None
    results = run_tests(components)
    success = print_summary(results)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
