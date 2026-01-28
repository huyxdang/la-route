"""
Test script for Le-Route backend.
Tests document ingestion and Q&A with different query types.
"""

import os
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from models import DocType
from embeddings import EmbeddingsManager
from router import RuleBasedRouter
from prompts import build_qa_prompt, get_system_prompt
from mistralai import Mistral


# Sample policy document for testing
SAMPLE_DOCUMENT = """
ACME Corporation Employee Handbook - Policies and Procedures

Section 1: Vacation and Time Off Policy

All full-time employees are entitled to 15 days of paid time off (PTO) per calendar year. PTO accrues at a rate of 1.25 days per month worked. Unused PTO may be carried over to the following year, up to a maximum of 5 days.

Vacation requests must be submitted at least 2 weeks in advance through the HR portal. Requests are approved based on business needs and team coverage. During peak business periods (Q4), vacation requests may be limited.

Section 2: Remote Work Policy

Employees may work remotely up to 3 days per week with manager approval. Remote work arrangements must be documented in writing. Employees must maintain core hours of 10 AM to 3 PM in their local timezone for team collaboration.

All remote workers must have a secure, dedicated workspace and reliable internet connection (minimum 25 Mbps). Company equipment must be used on secure networks only.

Section 3: Expense Reimbursement

Business expenses must be submitted within 30 days of the expense date. Receipts are required for all expenses over $25. The following categories are eligible for reimbursement:
- Travel (flights, hotels, ground transportation)
- Meals during business travel (up to $75/day)
- Professional development and conferences (pre-approved)
- Home office equipment (up to $500/year)

Expense reports are processed weekly on Fridays. Direct deposit reimbursements typically arrive within 5 business days.

Section 4: Termination Procedures

Employment at ACME Corporation is at-will. Either party may terminate the employment relationship at any time, with or without cause. However, voluntary resignations require 2 weeks' written notice.

Upon termination, employees must return all company property including laptops, badges, and any confidential materials. Final paychecks are issued within the timeframe required by applicable state law.

Employees terminated for cause may be ineligible for rehire. Termination for gross misconduct, including but not limited to theft, harassment, or violation of confidentiality agreements, will result in immediate dismissal without severance.

Section 5: Social Media Policy

Employees are prohibited from sharing confidential company information on social media platforms. This includes but is not limited to: financial data, product roadmaps, customer information, and internal communications.

Personal social media use during work hours should be limited and must not interfere with job responsibilities. Employees who publicly identify themselves as ACME employees must include a disclaimer that views expressed are their own.

Violation of the social media policy may result in disciplinary action up to and including termination.
"""

SIMPLE_QUERIES = [
    "How many vacation days do employees get?",
    "What is the expense deadline?",
    "How many days per week can employees work remotely?"
]

COMPLEX_QUERIES = [
    "Can the company terminate an employee for posting on social media?",
    "What are all the requirements for working remotely?",
    "Explain the relationship between termination and confidentiality agreements."
]


def test_chunking():
    """Test the text chunking functionality."""
    print("\n" + "="*60)
    print("TEST: Text Chunking")
    print("="*60)
    
    manager = EmbeddingsManager()
    chunks = manager._chunk_text(SAMPLE_DOCUMENT)
    
    print(f"Document length: {len(SAMPLE_DOCUMENT)} characters")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Estimated tokens: {manager._estimate_tokens(SAMPLE_DOCUMENT)}")
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} ({len(chunk)} chars, ~{manager._estimate_tokens(chunk)} tokens):")
        print(f"  Preview: {chunk[:100]}...")
    
    assert len(chunks) > 0, "Should create at least one chunk"
    print("\n✓ Chunking test PASSED")


def test_rule_based_router():
    """Test the rule-based router."""
    print("\n" + "="*60)
    print("TEST: Rule-Based Router")
    print("="*60)
    
    router = RuleBasedRouter()
    
    # Test simple query
    decision = router.route(
        query="How many vacation days do employees get?",
        doc_type=DocType.POLICY,
        top_similarity=0.85,
        num_relevant_chunks=1
    )
    print(f"\nSimple query: {decision}")
    assert decision.model.value == "ministral-8b-latest", "Simple query should route to small model"
    
    # Test complex/risky query
    decision = router.route(
        query="Can we terminate an employee for social media violations?",
        doc_type=DocType.POLICY,
        top_similarity=0.7,
        num_relevant_chunks=3
    )
    print(f"Complex query: {decision}")
    assert decision.model.value == "mistral-large-latest", "Complex/risky query should route to large model"
    
    print("\n✓ Router test PASSED")


def test_full_pipeline():
    """Test the full document Q&A pipeline."""
    print("\n" + "="*60)
    print("TEST: Full Pipeline (Ingest + Q&A)")
    print("="*60)
    
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("⚠ MISTRAL_API_KEY not set, skipping API tests")
        return
    
    # Initialize
    manager = EmbeddingsManager(api_key=api_key)
    router = RuleBasedRouter()
    client = Mistral(api_key=api_key)
    
    # Ingest document
    print("\nIngesting document...")
    start = time.time()
    store = manager.ingest_document(
        text=SAMPLE_DOCUMENT,
        doc_type=DocType.POLICY,
        title="ACME Employee Handbook"
    )
    print(f"  Session ID: {store.session_id}")
    print(f"  Chunks created: {len(store.chunks)}")
    print(f"  Total tokens: {store.total_tokens}")
    print(f"  Ingestion time: {time.time() - start:.2f}s")
    
    # Test simple query
    print("\n--- Simple Query ---")
    query = "How many vacation days do employees get?"
    print(f"Q: {query}")
    
    start = time.time()
    chunks, similarities, query_emb = manager.retrieve_chunks(
        session_id=store.session_id,
        query=query,
        top_k=3
    )
    
    decision = router.route(
        query=query,
        doc_type=store.doc_type,
        top_similarity=float(similarities[0]),
        num_relevant_chunks=int(sum(similarities > 0.5))
    )
    print(f"Routing: {decision.model.value} ({decision.reason})")
    
    # Generate answer
    chunk_data = [{"id": c.id, "text": c.text} for c in chunks]
    user_prompt = build_qa_prompt(chunk_data, query)
    
    response = client.chat.complete(
        model=decision.model.value,
        messages=[
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=500
    )
    
    answer = response.choices[0].message.content
    print(f"A: {answer}")
    print(f"Time: {time.time() - start:.2f}s")
    print(f"Model: {decision.model.value}")
    
    # Test complex query
    print("\n--- Complex Query ---")
    query = "Can the company terminate an employee for posting on social media? What are the implications?"
    print(f"Q: {query}")
    
    start = time.time()
    chunks, similarities, query_emb = manager.retrieve_chunks(
        session_id=store.session_id,
        query=query,
        top_k=5
    )
    
    decision = router.route(
        query=query,
        doc_type=store.doc_type,
        top_similarity=float(similarities[0]),
        num_relevant_chunks=int(sum(similarities > 0.5))
    )
    print(f"Routing: {decision.model.value} ({decision.reason})")
    
    chunk_data = [{"id": c.id, "text": c.text} for c in chunks]
    user_prompt = build_qa_prompt(chunk_data, query)
    
    response = client.chat.complete(
        model=decision.model.value,
        messages=[
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=500
    )
    
    answer = response.choices[0].message.content
    print(f"A: {answer}")
    print(f"Time: {time.time() - start:.2f}s")
    print(f"Model: {decision.model.value}")
    
    print("\n✓ Full pipeline test PASSED")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("LE-ROUTE BACKEND TESTS")
    print("="*60)
    
    test_chunking()
    test_rule_based_router()
    test_full_pipeline()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()
