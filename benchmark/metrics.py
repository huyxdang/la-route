"""
Benchmark metrics for Le-Route evaluation.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BenchmarkResults:
    """Container for benchmark results."""
    # Answer quality
    answer_accuracy: float = 0.0
    answer_f1: float = 0.0
    
    # Citation quality
    citation_precision: float = 0.0
    citation_recall: float = 0.0
    
    # Abstention
    abstention_accuracy: float = 0.0
    
    # Routing
    routing_accuracy: float = 0.0
    small_model_usage_pct: float = 0.0
    
    # Cost
    total_cost_usd: float = 0.0
    baseline_cost_usd: float = 0.0
    cost_savings_pct: float = 0.0
    
    # Latency
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    
    # Counts
    total_queries: int = 0
    small_model_queries: int = 0
    large_model_queries: int = 0
    
    # Details
    per_query_results: list = field(default_factory=list)


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text


def compute_word_overlap(pred: str, truth: str) -> tuple[float, float, float]:
    """
    Compute precision, recall, F1 based on word overlap.
    
    Returns:
        (precision, recall, f1)
    """
    pred_words = set(normalize_text(pred).split())
    truth_words = set(normalize_text(truth).split())
    
    if not pred_words and not truth_words:
        return 1.0, 1.0, 1.0
    if not pred_words:
        return 0.0, 0.0, 0.0
    if not truth_words:
        return 0.0, 0.0, 0.0
    
    overlap = len(pred_words & truth_words)
    precision = overlap / len(pred_words)
    recall = overlap / len(truth_words)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


def answer_accuracy(predicted: str, ground_truth: str, threshold: float = 0.5) -> float:
    """
    Compute answer accuracy using word overlap.
    
    Args:
        predicted: Model's answer
        ground_truth: Expected answer
        threshold: Minimum F1 score to consider correct
    
    Returns:
        1.0 if correct, 0.0 otherwise
    """
    if not predicted or not ground_truth:
        return 0.0
    
    # Check exact match
    if normalize_text(predicted) == normalize_text(ground_truth):
        return 1.0
    
    # Check containment
    pred_norm = normalize_text(predicted)
    truth_norm = normalize_text(ground_truth)
    if truth_norm in pred_norm or pred_norm in truth_norm:
        return 1.0
    
    # Check word overlap
    _, _, f1 = compute_word_overlap(predicted, ground_truth)
    return 1.0 if f1 >= threshold else 0.0


def citation_accuracy(
    cited_chunk_ids: list[int],
    relevant_chunk_ids: list[int]
) -> tuple[float, float]:
    """
    Compute citation precision and recall.
    
    Args:
        cited_chunk_ids: Chunk IDs cited by the model
        relevant_chunk_ids: Ground truth relevant chunk IDs
    
    Returns:
        (precision, recall)
    """
    if not cited_chunk_ids:
        return 0.0, 0.0 if relevant_chunk_ids else 1.0
    if not relevant_chunk_ids:
        return 0.0, 1.0
    
    cited_set = set(cited_chunk_ids)
    relevant_set = set(relevant_chunk_ids)
    
    correct_citations = len(cited_set & relevant_set)
    
    precision = correct_citations / len(cited_set) if cited_set else 0.0
    recall = correct_citations / len(relevant_set) if relevant_set else 0.0
    
    return precision, recall


def abstention_accuracy(
    abstained: bool,
    answer_in_document: bool
) -> float:
    """
    Check if abstention decision was correct.
    
    Args:
        abstained: Whether model abstained
        answer_in_document: Whether answer exists in document
    
    Returns:
        1.0 if correct decision, 0.0 otherwise
    """
    # Should abstain if answer not in document
    # Should not abstain if answer is in document
    should_abstain = not answer_in_document
    return 1.0 if abstained == should_abstain else 0.0


def routing_accuracy(
    model_used: str,
    small_correct: bool,
    large_correct: bool
) -> float:
    """
    Check if routing decision was optimal.
    
    Optimal routing:
    - Use small if small model would be correct
    - Use large if only large would be correct
    
    Returns:
        1.0 if optimal, 0.0 otherwise
    """
    used_small = "8b" in model_used.lower() or "ministral" in model_used.lower()
    
    if small_correct:
        # Small is sufficient - should use small
        return 1.0 if used_small else 0.0
    else:
        # Need large - should use large
        return 0.0 if used_small else 1.0


def cost_estimate(
    model: str,
    input_tokens: int,
    output_tokens: int
) -> float:
    """
    Estimate cost in USD.
    
    Pricing (per 1M tokens):
    - ministral-8b-latest: $0.10 input, $0.10 output
    - mistral-large-latest: $2.00 input, $6.00 output
    """
    if "8b" in model.lower() or "ministral" in model.lower():
        input_rate = 0.10
        output_rate = 0.10
    else:  # large
        input_rate = 2.00
        output_rate = 6.00
    
    input_cost = (input_tokens / 1_000_000) * input_rate
    output_cost = (output_tokens / 1_000_000) * output_rate
    
    return input_cost + output_cost


def aggregate_results(per_query_results: list[dict]) -> BenchmarkResults:
    """
    Aggregate per-query results into overall metrics.
    """
    if not per_query_results:
        return BenchmarkResults()
    
    n = len(per_query_results)
    
    # Calculate averages
    results = BenchmarkResults(
        total_queries=n,
        per_query_results=per_query_results
    )
    
    # Answer accuracy
    results.answer_accuracy = sum(r.get("answer_correct", 0) for r in per_query_results) / n
    
    # Average F1
    f1_scores = [r.get("answer_f1", 0) for r in per_query_results]
    results.answer_f1 = sum(f1_scores) / n
    
    # Citation metrics
    precisions = [r.get("citation_precision", 0) for r in per_query_results]
    recalls = [r.get("citation_recall", 0) for r in per_query_results]
    results.citation_precision = sum(precisions) / n
    results.citation_recall = sum(recalls) / n
    
    # Abstention accuracy
    abstention_scores = [r.get("abstention_correct", 0) for r in per_query_results if "abstention_correct" in r]
    results.abstention_accuracy = sum(abstention_scores) / len(abstention_scores) if abstention_scores else 0
    
    # Routing accuracy
    routing_scores = [r.get("routing_correct", 0) for r in per_query_results if "routing_correct" in r]
    results.routing_accuracy = sum(routing_scores) / len(routing_scores) if routing_scores else 0
    
    # Model usage
    small_count = sum(1 for r in per_query_results if "8b" in r.get("model_used", "").lower())
    results.small_model_queries = small_count
    results.large_model_queries = n - small_count
    results.small_model_usage_pct = (small_count / n) * 100
    
    # Cost
    results.total_cost_usd = sum(r.get("cost", 0) for r in per_query_results)
    results.baseline_cost_usd = sum(r.get("baseline_cost", 0) for r in per_query_results)
    if results.baseline_cost_usd > 0:
        results.cost_savings_pct = ((results.baseline_cost_usd - results.total_cost_usd) / 
                                     results.baseline_cost_usd) * 100
    
    # Latency
    latencies = [r.get("latency_ms", 0) for r in per_query_results]
    results.avg_latency_ms = sum(latencies) / n
    sorted_latencies = sorted(latencies)
    p95_idx = int(n * 0.95)
    results.p95_latency_ms = sorted_latencies[p95_idx] if p95_idx < n else sorted_latencies[-1]
    
    return results


def print_results(results: BenchmarkResults):
    """Pretty print benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    
    print(f"\n📊 Query Statistics")
    print(f"  Total queries: {results.total_queries}")
    print(f"  Small model: {results.small_model_queries} ({results.small_model_usage_pct:.1f}%)")
    print(f"  Large model: {results.large_model_queries}")
    
    print(f"\n📝 Answer Quality")
    print(f"  Accuracy: {results.answer_accuracy*100:.1f}%")
    print(f"  Average F1: {results.answer_f1:.3f}")
    
    print(f"\n📑 Citation Quality")
    print(f"  Precision: {results.citation_precision*100:.1f}%")
    print(f"  Recall: {results.citation_recall*100:.1f}%")
    
    if results.abstention_accuracy > 0:
        print(f"\n🚫 Abstention")
        print(f"  Accuracy: {results.abstention_accuracy*100:.1f}%")
    
    print(f"\n🔀 Routing")
    print(f"  Accuracy: {results.routing_accuracy*100:.1f}%")
    
    print(f"\n💰 Cost Analysis")
    print(f"  Total cost: ${results.total_cost_usd:.4f}")
    print(f"  Baseline (always large): ${results.baseline_cost_usd:.4f}")
    print(f"  Savings: {results.cost_savings_pct:.1f}%")
    
    print(f"\n⏱ Latency")
    print(f"  Average: {results.avg_latency_ms:.0f}ms")
    print(f"  P95: {results.p95_latency_ms:.0f}ms")
    
    print("\n" + "=" * 60)
