"""
Generate routing labels by running both models on each example.
Label = 0 (Small) if small model got it correct
Label = 1 (Large) if only large model got it correct, or both failed
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from dotenv import load_dotenv
from mistralai import Mistral

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.prompts import SYSTEM_PROMPT, build_qa_prompt

load_dotenv()

# Models
MODEL_SMALL = "ministral-8b-latest"
MODEL_LARGE = "mistral-large-latest"
EMBED_MODEL = "mistral-embed"

# Rate limiting
REQUESTS_PER_MINUTE = 30
DELAY_BETWEEN_REQUESTS = 60.0 / REQUESTS_PER_MINUTE


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    import re
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text


def compute_overlap(pred: str, truth: str) -> float:
    """Compute word overlap ratio between prediction and ground truth."""
    pred_words = set(normalize_text(pred).split())
    truth_words = set(normalize_text(truth).split())
    
    if not truth_words:
        return 0.0
    
    overlap = len(pred_words & truth_words)
    return overlap / len(truth_words)


def is_correct(prediction: str, ground_truth: str, threshold: float = 0.5) -> bool:
    """
    Check if prediction is correct.
    Uses word overlap with threshold.
    """
    if not prediction or not ground_truth:
        return False
    
    # Check for exact match after normalization
    if normalize_text(prediction) == normalize_text(ground_truth):
        return True
    
    # Check for containment
    pred_norm = normalize_text(prediction)
    truth_norm = normalize_text(ground_truth)
    
    if truth_norm in pred_norm:
        return True
    
    # Check word overlap
    overlap = compute_overlap(prediction, ground_truth)
    return overlap >= threshold


def generate_answer(
    client: Mistral,
    model: str,
    question: str,
    chunks: list[dict],
    max_retries: int = 3
) -> tuple[str, int, int]:
    """
    Generate answer using specified model.
    Returns (answer, input_tokens, output_tokens)
    """
    # Build prompt
    chunks_for_prompt = [
        (c["text"], 0.9, c["chunk_id"])  # Use placeholder relevance
        for c in chunks[:5]  # Top 5 chunks
    ]
    user_prompt = build_qa_prompt(question, chunks_for_prompt)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.complete(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=512
            )
            
            return (
                response.choices[0].message.content,
                response.usage.prompt_tokens,
                response.usage.completion_tokens
            )
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                print(f"\n⚠ API error ({model}): {e}, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"\n✗ Failed after {max_retries} attempts: {e}")
                return "", 0, 0
    
    return "", 0, 0


def embed_texts(client: Mistral, texts: list[str]) -> list[list[float]]:
    """Generate embeddings for texts."""
    if not texts:
        return []
    
    response = client.embeddings.create(
        model=EMBED_MODEL,
        inputs=texts
    )
    
    return [item.embedding for item in response.data]


def generate_labels(
    input_dir: str = "data/processed",
    output_dir: str = "data/labeled",
    max_examples: Optional[int] = None,
    resume: bool = True
):
    """
    Generate routing labels for all examples.
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not set")
    
    client = Mistral(api_key=api_key)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split in ["train", "validation"]:
        input_file = input_path / f"{split}.jsonl"
        output_file = output_path / f"{split}.jsonl"
        
        if not input_file.exists():
            print(f"⚠ Skipping {split} - input file not found")
            continue
        
        print(f"\n{'='*50}")
        print(f"Processing {split} split")
        print(f"{'='*50}")
        
        # Load existing progress if resuming
        processed_ids = set()
        existing_data = []
        if resume and output_file.exists():
            with open(output_file, "r") as f:
                for line in f:
                    item = json.loads(line)
                    processed_ids.add(item["id"])
                    existing_data.append(item)
            print(f"Resuming from {len(processed_ids)} previously processed examples")
        
        # Load input data
        with open(input_file, "r") as f:
            data = [json.loads(line) for line in f]
        
        if max_examples:
            data = data[:max_examples]
        
        # Filter already processed
        data_to_process = [d for d in data if d["id"] not in processed_ids]
        print(f"Processing {len(data_to_process)} new examples")
        
        # Stats
        stats = {
            "small_correct": 0,
            "large_correct": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "label_0_small": 0,
            "label_1_large": 0,
        }
        
        labeled_data = existing_data.copy()
        
        for item in tqdm(data_to_process, desc=f"Labeling {split}"):
            # Rate limiting
            time.sleep(DELAY_BETWEEN_REQUESTS)
            
            question = item["question"]
            chunks = item["chunks"]
            ground_truth = item["answer"]
            
            # Embed the question
            try:
                query_embedding = embed_texts(client, [question])[0]
            except Exception as e:
                print(f"\n⚠ Embedding failed for {item['id']}: {e}")
                continue
            
            # Embed chunks and compute similarities
            try:
                chunk_texts = [c["text"] for c in chunks[:5]]
                chunk_embeddings = embed_texts(client, chunk_texts)
                
                import numpy as np
                query_vec = np.array(query_embedding)
                query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-10)
                
                top_similarities = []
                for emb in chunk_embeddings:
                    emb_vec = np.array(emb)
                    emb_vec = emb_vec / (np.linalg.norm(emb_vec) + 1e-10)
                    sim = float(query_vec @ emb_vec)
                    top_similarities.append(sim)
            except Exception as e:
                print(f"\n⚠ Chunk embedding failed for {item['id']}: {e}")
                continue
            
            # Generate answer with small model
            small_answer, small_in, small_out = generate_answer(
                client, MODEL_SMALL, question, chunks
            )
            time.sleep(DELAY_BETWEEN_REQUESTS)
            
            # Generate answer with large model
            large_answer, large_in, large_out = generate_answer(
                client, MODEL_LARGE, question, chunks
            )
            
            # Evaluate correctness
            small_correct = is_correct(small_answer, ground_truth)
            large_correct = is_correct(large_answer, ground_truth)
            
            # Assign label
            # Label = 0 (Small) if small model got it correct
            # Label = 1 (Large) if only large got it correct, or both failed
            if small_correct:
                label = 0
                stats["label_0_small"] += 1
            else:
                label = 1
                stats["label_1_large"] += 1
            
            # Update stats
            if small_correct:
                stats["small_correct"] += 1
            if large_correct:
                stats["large_correct"] += 1
            if small_correct and large_correct:
                stats["both_correct"] += 1
            if not small_correct and not large_correct:
                stats["both_wrong"] += 1
            
            # Store labeled example
            labeled_item = {
                "id": item["id"],
                "question": question,
                "query_embedding": query_embedding,
                "top_similarities": top_similarities,
                "num_relevant_chunks": sum(1 for s in top_similarities if s > 0.5),
                "query_token_count": len(question.split()),
                "doc_token_count": sum(c.get("tokens_approx", 100) for c in chunks),
                "doc_type": item["doc_type"],
                "small_answer": small_answer,
                "large_answer": large_answer,
                "ground_truth": ground_truth,
                "small_correct": small_correct,
                "large_correct": large_correct,
                "label": label,
            }
            labeled_data.append(labeled_item)
            
            # Save progress periodically
            if len(labeled_data) % 10 == 0:
                with open(output_file, "w") as f:
                    for ld in labeled_data:
                        f.write(json.dumps(ld) + "\n")
        
        # Final save
        with open(output_file, "w") as f:
            for ld in labeled_data:
                f.write(json.dumps(ld) + "\n")
        
        # Print statistics
        n = len(data_to_process)
        if n > 0:
            print(f"\n=== {split} Labeling Statistics ===")
            print(f"Total processed: {n}")
            print(f"Small model correct: {stats['small_correct']} ({stats['small_correct']/n*100:.1f}%)")
            print(f"Large model correct: {stats['large_correct']} ({stats['large_correct']/n*100:.1f}%)")
            print(f"Both correct: {stats['both_correct']} ({stats['both_correct']/n*100:.1f}%)")
            print(f"Both wrong: {stats['both_wrong']} ({stats['both_wrong']/n*100:.1f}%)")
            print(f"Label 0 (use Small): {stats['label_0_small']} ({stats['label_0_small']/n*100:.1f}%)")
            print(f"Label 1 (use Large): {stats['label_1_large']} ({stats['label_1_large']/n*100:.1f}%)")
        
        print(f"✓ Saved {len(labeled_data)} examples to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate routing labels")
    parser.add_argument("--max", type=int, default=None, help="Max examples to process")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh")
    args = parser.parse_args()
    
    generate_labels(max_examples=args.max, resume=not args.no_resume)
