"""
Analyze DocBench evaluation results by domain and question type.

Usage:
    python eval_result.py --input data/ministral-8b_results.eval.jsonl
    python eval_result.py --input data/mistral-large_results.eval.jsonl
    python eval_result.py --compare data/ministral-8b_results.eval.jsonl data/mistral-large_results.eval.jsonl
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


# Question type mappings
QUESTION_TYPES = {
    'text-only': 'text',
    'multimodal-f': 'multimodal',
    'multimodal-t': 'multimodal', 
    'multimodal': 'multimodal',
    'meta-data': 'metadata',
}

# Document domain ranges (folder numbers)
DOMAINS = {
    'academic': range(0, 49),
    'finance': range(49, 89),
    'government': range(89, 133),
    'legal': range(133, 179),
    'news': range(179, 229),
}


def load_results(filepath: str) -> list[dict]:
    """Load evaluation results from JSONL file."""
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def get_domain(folder_num: int) -> str:
    """Get domain name from folder number."""
    for domain, folder_range in DOMAINS.items():
        if folder_num in folder_range:
            return domain
    return 'unknown'


def analyze_results(results: list[dict]) -> dict:
    """Analyze results by domain and question type."""
    # Initialize counters
    by_domain = defaultdict(lambda: {'correct': 0, 'total': 0})
    by_type = defaultdict(lambda: {'correct': 0, 'total': 0})
    overall = {'correct': 0, 'total': 0}
    
    for r in results:
        score = r.get('eval_score', 0)
        folder = int(r.get('file', 0))
        qtype = QUESTION_TYPES.get(r.get('type', ''), 'unknown')
        domain = get_domain(folder)
        
        # Update counters
        overall['total'] += 1
        overall['correct'] += score
        
        by_domain[domain]['total'] += 1
        by_domain[domain]['correct'] += score
        
        by_type[qtype]['total'] += 1
        by_type[qtype]['correct'] += score
    
    return {
        'overall': overall,
        'by_domain': dict(by_domain),
        'by_type': dict(by_type),
    }


def print_analysis(analysis: dict, name: str = "Results"):
    """Print formatted analysis."""
    print(f"\n{'=' * 60}")
    print(f" {name}")
    print('=' * 60)
    
    # Overall
    overall = analysis['overall']
    acc = overall['correct'] / overall['total'] * 100 if overall['total'] > 0 else 0
    print(f"\nOverall: {overall['correct']}/{overall['total']} ({acc:.1f}%)")
    
    # By domain
    print(f"\n{'By Domain':-^40}")
    for domain in ['academic', 'finance', 'government', 'legal', 'news']:
        stats = analysis['by_domain'].get(domain, {'correct': 0, 'total': 0})
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total'] * 100
            print(f"  {domain.capitalize():12} {stats['correct']:3}/{stats['total']:3} ({acc:5.1f}%)")
    
    # By question type
    print(f"\n{'By Question Type':-^40}")
    for qtype in ['text', 'multimodal', 'metadata']:
        stats = analysis['by_type'].get(qtype, {'correct': 0, 'total': 0})
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total'] * 100
            print(f"  {qtype.capitalize():12} {stats['correct']:3}/{stats['total']:3} ({acc:5.1f}%)")
    
    print()


def compare_models(file1: str, file2: str):
    """Compare two model results side by side."""
    results1 = load_results(file1)
    results2 = load_results(file2)
    
    name1 = Path(file1).stem.replace('_results.eval', '').replace('_', ' ').title()
    name2 = Path(file2).stem.replace('_results.eval', '').replace('_', ' ').title()
    
    analysis1 = analyze_results(results1)
    analysis2 = analyze_results(results2)
    
    print(f"\n{'=' * 60}")
    print(f" Model Comparison: {name1} vs {name2}")
    print('=' * 60)
    
    # Overall comparison
    acc1 = analysis1['overall']['correct'] / analysis1['overall']['total'] * 100
    acc2 = analysis2['overall']['correct'] / analysis2['overall']['total'] * 100
    print(f"\nOverall Accuracy:")
    print(f"  {name1:20} {acc1:5.1f}%")
    print(f"  {name2:20} {acc2:5.1f}%")
    print(f"  {'Difference':20} {acc2 - acc1:+5.1f}%")
    
    # Domain comparison
    print(f"\n{'By Domain':-^50}")
    print(f"  {'Domain':12} {name1:>12} {name2:>12} {'Diff':>8}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*8}")
    for domain in ['academic', 'finance', 'government', 'legal', 'news']:
        s1 = analysis1['by_domain'].get(domain, {'correct': 0, 'total': 0})
        s2 = analysis2['by_domain'].get(domain, {'correct': 0, 'total': 0})
        if s1['total'] > 0:
            a1 = s1['correct'] / s1['total'] * 100
            a2 = s2['correct'] / s2['total'] * 100
            print(f"  {domain.capitalize():12} {a1:11.1f}% {a2:11.1f}% {a2-a1:+7.1f}%")
    
    # Type comparison
    print(f"\n{'By Question Type':-^50}")
    print(f"  {'Type':12} {name1:>12} {name2:>12} {'Diff':>8}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*8}")
    for qtype in ['text', 'multimodal', 'metadata']:
        s1 = analysis1['by_type'].get(qtype, {'correct': 0, 'total': 0})
        s2 = analysis2['by_type'].get(qtype, {'correct': 0, 'total': 0})
        if s1['total'] > 0:
            a1 = s1['correct'] / s1['total'] * 100
            a2 = s2['correct'] / s2['total'] * 100
            print(f"  {qtype.capitalize():12} {a1:11.1f}% {a2:11.1f}% {a2-a1:+7.1f}%")
    
    # Routing analysis
    print(f"\n{'Routing Analysis':-^50}")
    both_correct = 0
    only_small = 0
    only_large = 0
    both_wrong = 0
    
    for r1, r2 in zip(results1, results2):
        s1 = r1.get('eval_score', 0)
        s2 = r2.get('eval_score', 0)
        if s1 and s2:
            both_correct += 1
        elif s1 and not s2:
            only_small += 1
        elif not s1 and s2:
            only_large += 1
        else:
            both_wrong += 1
    
    total = len(results1)
    print(f"  Both correct (use small):  {both_correct:4} ({both_correct/total*100:5.1f}%)")
    print(f"  Only {name1} correct:      {only_small:4} ({only_small/total*100:5.1f}%)")
    print(f"  Only {name2} correct:      {only_large:4} ({only_large/total*100:5.1f}%)")
    print(f"  Both wrong:                {both_wrong:4} ({both_wrong/total*100:5.1f}%)")
    print(f"\n  Routing potential: {both_correct/total*100:.1f}% of queries can use cheap model")
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze DocBench evaluation results")
    parser.add_argument(
        "--input",
        type=str,
        help="Input eval JSONL file to analyze"
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=('FILE1', 'FILE2'),
        help="Compare two eval files (e.g., small model vs large model)"
    )
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models(args.compare[0], args.compare[1])
    elif args.input:
        results = load_results(args.input)
        name = Path(args.input).stem.replace('_results.eval', '').replace('_', ' ').title()
        analysis = analyze_results(results)
        print_analysis(analysis, name)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
