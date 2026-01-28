"""
Process DocBench data into chunks and prepare for label generation.
"""

import os
import json
from pathlib import Path
from typing import Optional
from tqdm import tqdm

# Chunking parameters (matching backend)
CHUNK_SIZE = 500  # tokens (approximate)
CHUNK_OVERLAP = 100  # tokens
CHARS_PER_TOKEN = 4


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split text into overlapping chunks.
    Returns list of chunk dicts with text and positions.
    """
    chunk_chars = chunk_size * CHARS_PER_TOKEN
    overlap_chars = overlap * CHARS_PER_TOKEN
    
    chunks = []
    text = text.strip()
    
    if not text:
        return chunks
    
    if len(text) <= chunk_chars:
        return [{
            "chunk_id": 0,
            "text": text,
            "start_char": 0,
            "end_char": len(text),
            "tokens_approx": len(text) // CHARS_PER_TOKEN
        }]
    
    start = 0
    chunk_id = 0
    
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        
        # Try to break at sentence boundary
        if end < len(text):
            search_start = start + int(chunk_chars * 0.8)
            search_region = text[search_start:end]
            
            for sep in ['. ', '.\n', '? ', '!\n', '! ', '?\n']:
                last_sep = search_region.rfind(sep)
                if last_sep != -1:
                    end = search_start + last_sep + len(sep)
                    break
        
        chunk_text = text[start:end].strip()
        
        if chunk_text:
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "start_char": start,
                "end_char": end,
                "tokens_approx": len(chunk_text) // CHARS_PER_TOKEN
            })
            chunk_id += 1
        
        start = end - overlap_chars
        if start >= len(text) - overlap_chars:
            break
    
    return chunks


def infer_doc_type(text: str, existing_type: Optional[str] = None) -> str:
    """
    Infer document type from content using keyword matching.
    """
    if existing_type and existing_type in ["policy", "contract", "legal", "technical", "general"]:
        return existing_type
    
    text_lower = text.lower()
    
    # Legal indicators
    legal_keywords = ["whereas", "hereby", "jurisdiction", "arbitration", "litigation", 
                      "liability", "indemnify", "warrant", "covenant"]
    if sum(1 for k in legal_keywords if k in text_lower) >= 3:
        return "legal"
    
    # Contract indicators
    contract_keywords = ["agreement", "parties", "shall", "term", "terminate",
                         "payment", "services", "obligations", "breach"]
    if sum(1 for k in contract_keywords if k in text_lower) >= 3:
        return "contract"
    
    # Policy indicators
    policy_keywords = ["policy", "employee", "guidelines", "procedures", "handbook",
                       "compliance", "conduct", "responsible", "must"]
    if sum(1 for k in policy_keywords if k in text_lower) >= 3:
        return "policy"
    
    # Technical indicators
    technical_keywords = ["api", "endpoint", "request", "response", "authentication",
                          "parameter", "function", "method", "code", "error"]
    if sum(1 for k in technical_keywords if k in text_lower) >= 3:
        return "technical"
    
    return "general"


def find_relevant_chunks(chunks: list[dict], answer: str, question: str) -> list[int]:
    """
    Find which chunks contain information relevant to the answer.
    Uses simple text overlap heuristic.
    """
    if not answer:
        return []
    
    relevant_ids = []
    answer_lower = answer.lower()
    question_lower = question.lower()
    
    # Extract key terms from answer (longer words are more meaningful)
    answer_terms = set(word for word in answer_lower.split() if len(word) > 4)
    
    for chunk in chunks:
        chunk_lower = chunk["text"].lower()
        
        # Check for direct answer text overlap
        if len(answer) > 20 and answer_lower[:50] in chunk_lower:
            relevant_ids.append(chunk["chunk_id"])
            continue
        
        # Check for term overlap
        term_overlap = sum(1 for term in answer_terms if term in chunk_lower)
        if term_overlap >= 3 or (term_overlap >= 2 and len(answer_terms) <= 5):
            relevant_ids.append(chunk["chunk_id"])
    
    # If no relevant chunks found, use the first chunk as fallback
    if not relevant_ids and chunks:
        relevant_ids = [0]
    
    return relevant_ids


def process_docbench(
    input_dir: str = "data/raw/docbench",
    output_dir: str = "data/processed"
):
    """
    Process raw DocBench data into chunked format.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split_file in ["train.jsonl", "validation.jsonl"]:
        input_file = input_path / split_file
        if not input_file.exists():
            print(f"⚠ Skipping {split_file} - file not found")
            continue
        
        output_file = output_path / split_file
        
        print(f"\nProcessing {split_file}...")
        
        processed = []
        stats = {
            "total": 0,
            "avg_chunks": 0,
            "avg_doc_tokens": 0,
            "doc_types": {},
            "with_relevant_chunks": 0
        }
        
        with open(input_file, "r") as f:
            lines = f.readlines()
        
        for line in tqdm(lines, desc=f"Processing {split_file}"):
            item = json.loads(line)
            stats["total"] += 1
            
            # Get document text
            document = item.get("document", item.get("context", item.get("text", "")))
            question = item.get("question", item.get("query", ""))
            answer = item.get("answer", item.get("response", ""))
            
            if not document or not question:
                continue
            
            # Chunk the document
            chunks = chunk_text(document)
            stats["avg_chunks"] += len(chunks)
            stats["avg_doc_tokens"] += sum(c["tokens_approx"] for c in chunks)
            
            # Infer document type
            existing_type = item.get("doc_type", item.get("type"))
            doc_type = infer_doc_type(document, existing_type)
            stats["doc_types"][doc_type] = stats["doc_types"].get(doc_type, 0) + 1
            
            # Find relevant chunks
            relevant_chunk_ids = find_relevant_chunks(chunks, answer, question)
            if relevant_chunk_ids:
                stats["with_relevant_chunks"] += 1
            
            processed.append({
                "id": item.get("id", f"{split_file}_{stats['total']}"),
                "document": document,
                "question": question,
                "answer": answer,
                "doc_type": doc_type,
                "chunks": chunks,
                "relevant_chunk_ids": relevant_chunk_ids,
                "is_complex": item.get("is_complex", False),
            })
        
        # Save processed data
        with open(output_file, "w") as f:
            for item in processed:
                f.write(json.dumps(item) + "\n")
        
        # Print statistics
        n = stats["total"]
        print(f"\n=== {split_file} Statistics ===")
        print(f"Total examples: {n}")
        print(f"Avg chunks per doc: {stats['avg_chunks']/n:.1f}")
        print(f"Avg tokens per doc: {stats['avg_doc_tokens']/n:.0f}")
        print(f"Examples with relevant chunks: {stats['with_relevant_chunks']} ({stats['with_relevant_chunks']/n*100:.1f}%)")
        print(f"Doc types: {stats['doc_types']}")
        print(f"✓ Saved to {output_file}")


if __name__ == "__main__":
    process_docbench()
