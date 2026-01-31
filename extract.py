"""
Chunk + Embed papers with GROBID + mistral-embed

With each chunk, we store the following metadata:

- title: title of paper
- note_id: unique id of paper
- decision: decision of paper (oral, poster, etc.)
- keywords: keywords of paper (comma separated list)
- url: url of paper (openreview url)
- tldr: tldr of paper
"""

import requests
from lxml import etree
from pathlib import Path
import json
import re
import argparse
import os
from dotenv import load_dotenv
from mistralai import Mistral
import pandas as pd

load_dotenv()

NS = {"tei": "http://www.tei-c.org/ns/1.0"}
MISTRAL_EMBED_MODEL = "mistral-embed"
EMBED_BATCH_SIZE = 50
MAX_CHARS_PER_CHUNK = 16000  # ~4K tokens, safely under 8192 limit


def get_mistral_client() -> Mistral:
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY environment variable not set")
    return Mistral(api_key=api_key)


def embed_texts(client: Mistral, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=MISTRAL_EMBED_MODEL, inputs=texts)
    return [item.embedding for item in response.data]


def parse_grobid(pdf_path: str, grobid_url: str) -> list[dict]:
    with open(pdf_path, "rb") as f:
        resp = requests.post(f"{grobid_url}/api/processFulltextDocument", files={"input": f}, timeout=60)
    
    if resp.status_code != 200:
        print(f"  Failed: {resp.status_code}")
        return []
    
    root = etree.fromstring(resp.content)
    sections = []
    
    # Abstract
    abstract = root.find(".//tei:abstract", NS)
    if abstract is not None:
        text = clean_text(" ".join(abstract.itertext()))
        if text:
            sections.append({"section": "abstract", "content": text})
    
    # Body sections
    for div in root.findall(".//tei:body/tei:div", NS):
        head = div.find("tei:head", NS)
        section_name = head.text.strip().lower() if head is not None and head.text else "unnamed"
        paragraphs = div.findall(".//tei:p", NS)
        content = clean_text(" ".join(" ".join(p.itertext()) for p in paragraphs))
        
        if content and len(content) > 100:
            sections.append({"section": section_name, "content": content})
    
    return sections


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)
    return text.strip()


def split_long_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            current = sentence if len(sentence) <= max_chars else sentence[:max_chars]
    
    if current:
        chunks.append(current)
    
    return chunks or [text[:max_chars]]


def normalize_title(title: str) -> str:
    normalized = re.sub(r'[^a-z0-9\s\-]', '', title.lower().strip())
    return re.sub(r'\s+', '_', normalized)


def load_paper_metadata(csv_path: str) -> dict[str, dict]:
    df = pd.read_csv(csv_path)
    lookup = {}
    
    fields = {
        "title": "title",
        "note_id": "note_id",
        "decision": "decision",
        "keywords": "keywords",
        "url": "openreview_url",
        "tldr": "semanticscholar_tldr",
    }
    
    for _, row in df.iterrows():
        title = str(row.get("title", ""))
        if not title:
            continue
        
        metadata = {k: str(row.get(v, "")) if pd.notna(row.get(v)) else "" for k, v in fields.items()}
        lookup[normalize_title(title)] = metadata
    
    return lookup


def find_paper_metadata(pdf_stem: str, lookup: dict[str, dict]) -> dict:
    norm_stem = normalize_title(pdf_stem.replace("_", " "))
    
    if norm_stem in lookup:
        return lookup[norm_stem]
    
    for key, metadata in lookup.items():
        if key.startswith(norm_stem) or norm_stem.startswith(key):
            return metadata
    
    return {"title": pdf_stem.replace("_", " ")}


def load_processed_papers(output_path: str) -> set[str]:
    processed = set()
    if not Path(output_path).exists():
        return processed
    
    with open(output_path) as f:
        for line in f:
            try:
                chunk = json.loads(line)
                source = chunk.get("metadata", {}).get("source_file", "")
                if source:
                    processed.add(source)
            except json.JSONDecodeError:
                continue
    
    return processed


def check_grobid(url: str) -> bool:
    try:
        r = requests.get(f"{url}/api/isalive", timeout=5)
        return r.status_code == 200
    except requests.RequestException:
        return False


def main():
    parser = argparse.ArgumentParser(description="Extract paper sections via GROBID and embed with Mistral")
    parser.add_argument("--pdf-dir", type=str, default="papers/")
    parser.add_argument("--output", type=str, default="chunks.jsonl")
    parser.add_argument("-n", type=int, default=None)
    parser.add_argument("--grobid-url", type=str, default="http://localhost:8070")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-embed", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=MAX_CHARS_PER_CHUNK)
    parser.add_argument("--csv", type=str, default="neurips_2025.csv")
    args = parser.parse_args()
    
    grobid_url = args.grobid_url.rstrip("/")
    
    # Load metadata
    paper_lookup = {}
    if Path(args.csv).exists():
        paper_lookup = load_paper_metadata(args.csv)
        print(f"Loaded metadata for {len(paper_lookup)} papers")
    
    # Check dependencies
    if not check_grobid(grobid_url):
        print("GROBID not ready. Run: docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.8.0")
        return
    
    mistral_client = None
    if not args.no_embed:
        try:
            mistral_client = get_mistral_client()
        except ValueError as e:
            print(e)
            return
    
    # Get files to process
    processed = load_processed_papers(args.output) if not args.no_resume else set()
    if processed:
        print(f"Resuming: {len(processed)} papers already done")
    
    pdf_files = [p for p in Path(args.pdf_dir).glob("*.pdf") if p.stem not in processed]
    if args.n:
        pdf_files = pdf_files[:args.n]
    
    if not pdf_files:
        print("No new papers to process")
        return
    
    print(f"Processing {len(pdf_files)} papers...")
    
    mode = "a" if processed else "w"
    errors = []
    
    with open(args.output, mode) as f:
        for i, pdf_path in enumerate(pdf_files):
            print(f"[{i+1}/{len(pdf_files)}] {pdf_path.name}")
            
            try:
                metadata = find_paper_metadata(pdf_path.stem, paper_lookup)
                sections = parse_grobid(str(pdf_path), grobid_url)
                
                if not sections:
                    print("  → 0 sections (skipped)")
                    continue
                
                # Build chunks - use note_id for short, unique IDs
                chunks = []
                note_id = metadata.get("note_id", pdf_path.stem[:20])
                chunk_idx = 0
                for section in sections:
                    for part_idx, text_part in enumerate(split_long_text(section["content"], args.chunk_size)):
                        chunk_id = f"{note_id}_{chunk_idx:03d}"
                        chunk_idx += 1
                        
                        chunks.append({
                            "id": chunk_id,
                            "text": text_part,
                            "metadata": {**metadata, "source_file": pdf_path.stem, "section": section["section"]}
                        })
                
                # Embed
                if mistral_client:
                    texts = [c["text"] for c in chunks]
                    embeddings = []
                    for j in range(0, len(texts), EMBED_BATCH_SIZE):
                        embeddings.extend(embed_texts(mistral_client, texts[j:j + EMBED_BATCH_SIZE]))
                    for chunk, emb in zip(chunks, embeddings):
                        chunk["embedding"] = emb
                
                # Write
                for chunk in chunks:
                    f.write(json.dumps(chunk) + "\n")
                f.flush()
                
                print(f"  → {len(chunks)} chunks" + (" (embedded)" if mistral_client else ""))
            
            except Exception as e:
                print(f"  → ERROR: {e}")
                errors.append((pdf_path.name, str(e)))
                continue
    
    print(f"Done. Output: {args.output}")
    if errors:
        print(f"Errors: {len(errors)} papers failed")
        for name, err in errors[:10]:
            print(f"  - {name}: {err[:50]}...")


if __name__ == "__main__":
    main()