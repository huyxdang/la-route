"""
Training script for MLP router.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlp.model import RoutingMLP, save_model
from mlp.features import FeatureExtractor, extract_features_from_labeled_data


def load_labeled_data(path: str) -> list[dict]:
    """Load labeled data from JSONL file."""
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def prepare_dataset(
    data: list[dict],
    extractor: FeatureExtractor,
    fit_pca: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare features and labels from labeled data.
    
    Returns:
        (features, labels) arrays
    """
    # Extract embeddings for PCA fitting if needed
    if fit_pca:
        embeddings = np.array([d["query_embedding"] for d in data])
        extractor.fit_pca(embeddings)
    
    # Extract features
    features = []
    labels = []
    
    for item in tqdm(data, desc="Extracting features"):
        feat = extract_features_from_labeled_data(item, extractor)
        features.append(feat)
        labels.append(item["label"])
    
    return np.array(features), np.array(labels)


def train_model(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    patience: int = 5,
    device: str = "cpu"
) -> tuple[RoutingMLP, dict]:
    """
    Train the MLP router.
    
    Returns:
        (trained_model, training_log)
    """
    # Create data loaders
    train_dataset = TensorDataset(
        torch.tensor(train_features, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(val_features, dtype=torch.float32),
        torch.tensor(val_labels, dtype=torch.long)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = RoutingMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    
    log = {
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "best_epoch": 0
    }
    
    print(f"\nTraining MLP Router")
    print(f"Train: {len(train_features)}, Val: {len(val_features)}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    print("-" * 60)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(labels)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * len(labels)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        # Log
        log["epochs"].append(epoch + 1)
        log["train_loss"].append(train_loss)
        log["train_acc"].append(train_acc)
        log["val_loss"].append(val_loss)
        log["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            log["best_epoch"] = epoch + 1
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    print(f"\nBest model from epoch {log['best_epoch']} (val_loss: {best_val_loss:.4f})")
    
    return model, log


def main():
    parser = argparse.ArgumentParser(description="Train MLP router")
    parser.add_argument("--data-dir", default="data/labeled", help="Labeled data directory")
    parser.add_argument("--output-dir", default="mlp", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for labeled data
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "validation.jsonl"
    
    if not train_path.exists():
        print(f"✗ Training data not found at {train_path}")
        print("Run data/generate_labels.py first")
        return
    
    # Load data
    print("Loading labeled data...")
    train_data = load_labeled_data(train_path)
    
    if val_path.exists():
        val_data = load_labeled_data(val_path)
    else:
        # Split train data if no validation
        print("No validation data, using 80/20 split")
        split_idx = int(len(train_data) * 0.8)
        val_data = train_data[split_idx:]
        train_data = train_data[:split_idx]
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Initialize feature extractor and fit PCA
    extractor = FeatureExtractor()
    
    # Prepare datasets
    print("\nPreparing datasets...")
    train_features, train_labels = prepare_dataset(train_data, extractor, fit_pca=True)
    val_features, val_labels = prepare_dataset(val_data, extractor, fit_pca=False)
    
    # Print label distribution
    train_label_counts = np.bincount(train_labels)
    val_label_counts = np.bincount(val_labels)
    print(f"Train labels: 0={train_label_counts[0]}, 1={train_label_counts[1] if len(train_label_counts) > 1 else 0}")
    print(f"Val labels: 0={val_label_counts[0]}, 1={val_label_counts[1] if len(val_label_counts) > 1 else 0}")
    
    # Train model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model, log = train_model(
        train_features, train_labels,
        val_features, val_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        device=device
    )
    
    # Save model and PCA
    model_path = output_dir / "mlp_router.pt"
    pca_path = output_dir / "pca.pkl"
    log_path = output_dir / "training_log.json"
    
    save_model(model, str(model_path))
    extractor.save_pca(str(pca_path))
    
    # Save training log
    log["timestamp"] = datetime.now().isoformat()
    log["train_size"] = len(train_data)
    log["val_size"] = len(val_data)
    
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"✓ Training log saved to {log_path}")
    
    # Final stats
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best epoch: {log['best_epoch']}")
    print(f"Best val accuracy: {log['val_acc'][log['best_epoch']-1]:.3f}")
    print(f"Model saved to: {model_path}")
    print(f"PCA saved to: {pca_path}")


if __name__ == "__main__":
    main()
