"""
Upload labeled data to HuggingFace Hub.
"""

import os
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, create_repo

load_dotenv()

# Default dataset name
DEFAULT_DATASET_NAME = os.getenv("HF_DATASET_NAME", "le-route-routing-data")


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file."""
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def simplify_for_upload(data: list[dict]) -> list[dict]:
    """
    Simplify data for upload (remove large embeddings if needed).
    """
    simplified = []
    for item in data:
        # Keep essential fields
        simplified.append({
            "id": item["id"],
            "question": item["question"],
            "query_embedding": item["query_embedding"],  # Keep for training
            "top_similarities": item["top_similarities"],
            "num_relevant_chunks": item["num_relevant_chunks"],
            "query_token_count": item["query_token_count"],
            "doc_token_count": item["doc_token_count"],
            "doc_type": item["doc_type"],
            "ground_truth": item["ground_truth"],
            "small_correct": item["small_correct"],
            "large_correct": item["large_correct"],
            "label": item["label"],
        })
    return simplified


def upload_to_huggingface(
    data_dir: str = "data/labeled",
    dataset_name: str = DEFAULT_DATASET_NAME,
    private: bool = False
):
    """
    Upload labeled data to HuggingFace Hub.
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not set in environment")
    
    data_path = Path(data_dir)
    
    # Load data
    train_path = data_path / "train.jsonl"
    val_path = data_path / "validation.jsonl"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Train data not found at {train_path}")
    
    print("Loading data...")
    train_data = simplify_for_upload(load_jsonl(train_path))
    print(f"  Train: {len(train_data)} examples")
    
    if val_path.exists():
        val_data = simplify_for_upload(load_jsonl(val_path))
        print(f"  Validation: {len(val_data)} examples")
    else:
        val_data = None
        print("  No validation data")
    
    # Create datasets
    train_dataset = Dataset.from_list(train_data)
    
    if val_data:
        val_dataset = Dataset.from_list(val_data)
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset
        })
    else:
        dataset_dict = DatasetDict({"train": train_dataset})
    
    print(f"\nDataset info:")
    print(dataset_dict)
    
    # Create repo if it doesn't exist
    api = HfApi(token=hf_token)
    
    try:
        create_repo(
            repo_id=dataset_name,
            token=hf_token,
            repo_type="dataset",
            private=private,
            exist_ok=True
        )
        print(f"\n✓ Repository created/verified: {dataset_name}")
    except Exception as e:
        print(f"Note: {e}")
    
    # Push to hub
    print(f"\nUploading to HuggingFace Hub...")
    dataset_dict.push_to_hub(
        dataset_name,
        token=hf_token,
        private=private
    )
    
    print(f"\n✓ Dataset uploaded to: https://huggingface.co/datasets/{dataset_name}")
    
    # Create dataset card
    readme_content = f"""---
dataset_info:
  features:
    - name: id
      dtype: string
    - name: question
      dtype: string
    - name: query_embedding
      sequence: float64
    - name: top_similarities
      sequence: float64
    - name: num_relevant_chunks
      dtype: int64
    - name: query_token_count
      dtype: int64
    - name: doc_token_count
      dtype: int64
    - name: doc_type
      dtype: string
    - name: ground_truth
      dtype: string
    - name: small_correct
      dtype: bool
    - name: large_correct
      dtype: bool
    - name: label
      dtype: int64
  splits:
    - name: train
      num_examples: {len(train_data)}
    - name: validation
      num_examples: {len(val_data) if val_data else 0}
license: mit
---

# Le-Route Routing Dataset

Training data for intelligent model routing in document Q&A systems.

## Description

This dataset contains labeled examples for training an MLP router that decides whether to use:
- **Small model** (Ministral 8B, label=0): For simpler queries where the small model suffices
- **Large model** (Mistral Large, label=1): For complex queries requiring the larger model

## Labels

- `label = 0`: Small model got it correct (use small)
- `label = 1`: Only large model got it correct, or both failed (use large)

## Features

| Feature | Description |
|---------|-------------|
| `query_embedding` | 1024-dim embedding from mistral-embed |
| `top_similarities` | Top-5 chunk similarity scores |
| `num_relevant_chunks` | Chunks with similarity > 0.5 |
| `query_token_count` | Number of tokens in query |
| `doc_token_count` | Total tokens in document |
| `doc_type` | Document type (policy, contract, legal, technical, general) |

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{dataset_name}")
print(dataset)
```

## Project

Part of [Le-Route](https://github.com/your-username/le-route) - Enterprise Document Q&A with Intelligent Model Routing.
"""
    
    # Upload README
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=dataset_name,
        repo_type="dataset",
        token=hf_token
    )
    print("✓ Dataset card uploaded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload to HuggingFace")
    parser.add_argument("--name", default=DEFAULT_DATASET_NAME, help="Dataset name")
    parser.add_argument("--private", action="store_true", help="Make private")
    args = parser.parse_args()
    
    upload_to_huggingface(dataset_name=args.name, private=args.private)
