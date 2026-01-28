"""
Evaluation script for MLP router.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter

import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlp.model import load_model
from mlp.features import FeatureExtractor, extract_features_from_labeled_data


def load_labeled_data(path: str) -> list[dict]:
    """Load labeled data from JSONL file."""
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def evaluate_router(
    model,
    features: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.3,
    device: str = "cpu"
) -> dict:
    """
    Evaluate router on test data.
    
    Returns:
        Dict with evaluation metrics
    """
    model.eval()
    
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32).to(device)
        probs = model(x).cpu().numpy()
        
        # Predictions using threshold
        preds = (probs[:, 1] > threshold).astype(int)
    
    # Basic metrics
    correct = (preds == labels).sum()
    accuracy = correct / len(labels)
    
    # Per-class metrics
    tp = ((preds == 1) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    
    precision_large = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_large = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_large = 2 * precision_large * recall_large / (precision_large + recall_large) if (precision_large + recall_large) > 0 else 0
    
    precision_small = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_small = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_small = 2 * precision_small * recall_small / (precision_small + recall_small) if (precision_small + recall_small) > 0 else 0
    
    return {
        "accuracy": float(accuracy),
        "confusion_matrix": {
            "TP (use_large)": int(tp),
            "TN (use_small)": int(tn),
            "FP (should_use_small)": int(fp),
            "FN (should_use_large)": int(fn),
        },
        "class_0_small": {
            "precision": float(precision_small),
            "recall": float(recall_small),
            "f1": float(f1_small),
            "support": int(tn + fp)
        },
        "class_1_large": {
            "precision": float(precision_large),
            "recall": float(recall_large),
            "f1": float(f1_large),
            "support": int(tp + fn)
        },
        "routing_distribution": {
            "predicted_small": int((preds == 0).sum()),
            "predicted_large": int((preds == 1).sum()),
            "small_pct": float((preds == 0).sum() / len(preds) * 100),
        }
    }


def estimate_cost_savings(
    preds: np.ndarray,
    labels: np.ndarray,
    cost_small: float = 0.10,  # per 1M tokens
    cost_large: float = 2.00,   # per 1M tokens
    avg_tokens: int = 1000
) -> dict:
    """
    Estimate cost savings from routing.
    
    Returns:
        Dict with cost analysis
    """
    n = len(preds)
    
    # Baseline: always use large
    baseline_cost = n * cost_large * avg_tokens / 1_000_000
    
    # With routing
    routing_cost = 0
    correct_answers = 0
    
    for pred, label in zip(preds, labels):
        if pred == 0:  # Use small
            routing_cost += cost_small * avg_tokens / 1_000_000
            # Small model correct if label==0 (small is sufficient)
            if label == 0:
                correct_answers += 1
        else:  # Use large
            routing_cost += cost_large * avg_tokens / 1_000_000
            # Large model assumed correct
            correct_answers += 1
    
    savings = (baseline_cost - routing_cost) / baseline_cost * 100 if baseline_cost > 0 else 0
    accuracy_preservation = correct_answers / n * 100
    
    return {
        "baseline_cost_usd": round(baseline_cost, 4),
        "routing_cost_usd": round(routing_cost, 4),
        "savings_pct": round(savings, 1),
        "accuracy_with_routing_pct": round(accuracy_preservation, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate MLP router")
    parser.add_argument("--model", default="mlp/mlp_router.pt", help="Model path")
    parser.add_argument("--pca", default="mlp/pca.pkl", help="PCA path")
    parser.add_argument("--data", default="data/labeled/validation.jsonl", help="Validation data")
    parser.add_argument("--threshold", type=float, default=0.3, help="Routing threshold")
    parser.add_argument("--output", default="mlp/eval_results.json", help="Output path")
    args = parser.parse_args()
    
    # Load model
    if not Path(args.model).exists():
        print(f"✗ Model not found at {args.model}")
        return
    
    model = load_model(args.model)
    extractor = FeatureExtractor(args.pca)
    
    # Load data
    if not Path(args.data).exists():
        print(f"✗ Data not found at {args.data}")
        return
    
    data = load_labeled_data(args.data)
    print(f"Loaded {len(data)} examples")
    
    # Extract features
    print("Extracting features...")
    features = []
    labels = []
    
    for item in data:
        feat = extract_features_from_labeled_data(item, extractor)
        features.append(feat)
        labels.append(item["label"])
    
    features = np.array(features)
    labels = np.array(labels)
    
    print(f"Label distribution: {Counter(labels)}")
    
    # Evaluate
    print(f"\nEvaluating with threshold={args.threshold}...")
    results = evaluate_router(model, features, labels, threshold=args.threshold)
    
    # Get predictions for cost analysis
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32)
        probs = model(x).numpy()
        preds = (probs[:, 1] > args.threshold).astype(int)
    
    cost_analysis = estimate_cost_savings(preds, labels)
    results["cost_analysis"] = cost_analysis
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    print(f"\nAccuracy: {results['accuracy']:.3f}")
    
    print("\nConfusion Matrix:")
    cm = results["confusion_matrix"]
    print(f"  TP (correctly use large): {cm['TP (use_large)']}")
    print(f"  TN (correctly use small): {cm['TN (use_small)']}")
    print(f"  FP (should use small): {cm['FP (should_use_small)']}")
    print(f"  FN (should use large): {cm['FN (should_use_large)']}")
    
    print("\nClass Metrics:")
    print(f"  Small (0): P={results['class_0_small']['precision']:.3f}, "
          f"R={results['class_0_small']['recall']:.3f}, "
          f"F1={results['class_0_small']['f1']:.3f}")
    print(f"  Large (1): P={results['class_1_large']['precision']:.3f}, "
          f"R={results['class_1_large']['recall']:.3f}, "
          f"F1={results['class_1_large']['f1']:.3f}")
    
    print("\nRouting Distribution:")
    print(f"  Predicted Small: {results['routing_distribution']['predicted_small']} "
          f"({results['routing_distribution']['small_pct']:.1f}%)")
    print(f"  Predicted Large: {results['routing_distribution']['predicted_large']}")
    
    print("\nCost Analysis:")
    print(f"  Baseline (always large): ${cost_analysis['baseline_cost_usd']:.4f}")
    print(f"  With routing: ${cost_analysis['routing_cost_usd']:.4f}")
    print(f"  Savings: {cost_analysis['savings_pct']:.1f}%")
    print(f"  Accuracy preservation: {cost_analysis['accuracy_with_routing_pct']:.1f}%")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
