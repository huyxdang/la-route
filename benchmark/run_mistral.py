"""
Mistral API runner for DocBench benchmark.

Runs Mistral models on DocBench PDFs with full multimodal support (text + images).

Usage:
(1) Run the first time / fresh start:
    python run_mistral.py --model ministral-8b-latest --start 0 --end 228
    python run_mistral.py --model mistral-large-latest --start 0 --end 228 --no-images
    python run_mistral.py --model mistral-large-latest --start 0 --end 228 --workers 2 --no-resume

(2) Resume from previous run:
    python run_mistral.py --model ministral-8b-latest --start 0 --end 228 
    python run_mistral.py --model mistral-large-latest --start 0 --end 228 --workers 2

(3) Run by domain (smaller subsets):
    python run_mistral.py --model ministral-8b-latest --domain academic    # 49 docs
    python run_mistral.py --model ministral-8b-latest --domain finance    # 40 docs
    python run_mistral.py --model ministral-8b-latest --domain government # 44 docs
    python run_mistral.py --model ministral-8b-latest --domain legal      # 46 docs
    python run_mistral.py --model ministral-8b-latest --domain news       # 50 docs
"""

import json
import argparse
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
from mistralai import Mistral
from tenacity import retry, stop_after_attempt, wait_exponential

from pdf_extract import extract_pdf
from chunking import (
    estimate_doc_tokens, 
    SemanticChunker, 
    ChunkRetriever, 
    build_rag_content
)

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Mistral model mappings
MISTRAL_MODELS = {
    "ministral-8b": "ministral-8b-latest",
    "ministral-8b-latest": "ministral-8b-latest",
    "mistral-large": "mistral-large-latest",
    "mistral-large-latest": "mistral-large-latest",
}

# Cost per 1M tokens (as of Jan 2026)
MODEL_COSTS = {
    "ministral-8b-latest": {"input": 0.15, "output": 0.15},
    "mistral-large-latest": {"input": 0.5, "output": 1.5},
}

# Token threshold for switching to RAG (leaves room for images + response)
TOKEN_THRESHOLD = 150_000

# DocBench domain mappings (folder number ranges)
DOMAINS = {
    'academic': range(0, 49),      # 49 docs (folders 0-48)
    'finance': range(49, 89),      # 40 docs (folders 49-88)
    'government': range(89, 133),  # 44 docs (folders 89-132)
    'legal': range(133, 179),      # 46 docs (folders 133-178)
    'news': range(179, 229),       # 50 docs (folders 179-228)
}


def get_domain_folders(domain: str) -> list[int]:
    """Get list of folder numbers for a given domain."""
    if domain not in DOMAINS:
        raise ValueError(f"Unknown domain: {domain}. Choose from: {list(DOMAINS.keys())}")
    return list(DOMAINS[domain])


class MistralRunner:
    """Runs Mistral models on DocBench documents with multimodal support."""
    
    def __init__(self, model_name: str, max_images: int = 8, min_image_size: int = 200, workers: int = 1, include_images: bool = True):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment")
        
        self.client = Mistral(api_key=api_key)
        self.model = MISTRAL_MODELS.get(model_name, model_name)
        self.model_short = model_name.replace("-latest", "")
        self.max_images = max_images
        self.min_image_size = min_image_size
        self.workers = workers
        self.include_images = include_images
        
        # Semantic chunker for large documents (lazy init)
        self._chunker = None
        
        # Track costs (thread-safe)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._lock = threading.Lock()
        
        mode = "text+images" if include_images else "text-only"
        logger.info(f"Initialized MistralRunner with model: {self.model}, workers: {workers}, mode: {mode}")
    
    @property
    def chunker(self) -> SemanticChunker:
        """Lazy-init semantic chunker (makes API calls for embeddings)."""
        if self._chunker is None:
            self._chunker = SemanticChunker()
        return self._chunker
    
    def get_pdf_path(self, folder: str, data_dir: str) -> Path:
        """Get the PDF path for a folder."""
        folder_path = Path(data_dir) / str(folder)
        pdf_files = list(folder_path.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDF found in {folder_path}")
        return pdf_files[0]
    
    def get_document_content(self, folder: str, data_dir: str) -> list[dict]:
        """Extract text and images from PDF."""
        pdf_path = self.get_pdf_path(folder, data_dir)
        doc = extract_pdf(str(pdf_path), min_image_size=self.min_image_size, extract_images=self.include_images)
        
        content = []
        
        # Add document text with page markers
        text_parts = []
        for page in doc.pages:
            if page.text.strip():
                text_parts.append(f"[Page {page.page_num}]\n{page.text.strip()}")
        
        if text_parts:
            content.append({
                "type": "text",
                "text": "## Document Content:\n\n" + "\n\n---\n\n".join(text_parts)
            })
        
        # Add images with page annotations (only if enabled)
        if self.include_images:
            image_count = 0
            for page in doc.pages:
                for img in page.images:
                    if image_count >= self.max_images:
                        break
                    
                    content.append({
                        "type": "text",
                        "text": f"[Figure from page {img.page_num}]"
                    })
                    
                    mime_type = f"image/{img.image_type}"
                    if img.image_type == "jpg":
                        mime_type = "image/jpeg"
                    
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{img.image_b64}"
                        }
                    })
                    image_count += 1
                
                if image_count >= self.max_images:
                    break
            
            if image_count > 0:
                logger.info(f"  Included {image_count} image(s)")
        
        return content
    
    def get_questions(self, folder: str, data_dir: str) -> list[dict]:
        """Load questions from JSONL file."""
        folder_path = Path(data_dir) / str(folder)
        qa_path = folder_path / f"{folder}_qa.jsonl"
        
        if not qa_path.exists():
            raise FileNotFoundError(f"QA file not found: {qa_path}")
        
        questions = []
        with open(qa_path, 'r') as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
        
        return questions
    
    @retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def call_mistral(self, doc_content: list[dict], question: str, folder: str = "") -> tuple[str, int, int]:
        """Call Mistral API with document content (text + images)."""
        system_prompt = (
            "You are a helpful assistant that answers questions based on the given document. "
            "You can see both the text and any figures/charts from the document. "
            "Answer accurately and concisely based on all available information."
        )
        
        user_content = doc_content.copy()
        user_content.append({
            "type": "text",
            "text": f"\n## Question:\n{question}\n\nAnswer:"
        })
        
        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=1000,
                temperature=0.0
            )
        except Exception as e:
            # Log detailed error info
            logger.error(f"  API Error in folder {folder}: {type(e).__name__}: {e}")
            num_images = sum(1 for c in doc_content if c.get("type") == "image_url")
            total_bytes = sum(len(c.get("image_url", {}).get("url", "")) for c in doc_content if c.get("type") == "image_url")
            logger.error(f"  Request had {num_images} images, ~{total_bytes//1024}KB image data")
            raise
        
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, 'usage') and response.usage:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
        
        return response.choices[0].message.content.strip(), input_tokens, output_tokens
    
    def _update_tokens(self, input_tokens: int, output_tokens: int):
        """Thread-safe token counter update."""
        with self._lock:
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
    
    def run_folder(self, folder: str, data_dir: str, skip_questions: set[str] = None) -> list[dict]:
        """Run all questions for a single document folder.
        
        Args:
            folder: Folder number/name
            data_dir: Data directory path
            skip_questions: Set of question strings to skip (already completed)
        """
        if skip_questions is None:
            skip_questions = set()
        
        folder_path = Path(data_dir) / str(folder)
        
        if not folder_path.exists():
            logger.warning(f"Folder {folder} does not exist, skipping")
            return []
        
        try:
            questions = self.get_questions(folder, data_dir)
            pdf_path = self.get_pdf_path(folder, data_dir)
            doc = extract_pdf(str(pdf_path), min_image_size=self.min_image_size, extract_images=self.include_images)
        except FileNotFoundError as e:
            logger.error(f"Skipping folder {folder}: {e}")
            return []
        
        # Filter out already completed questions
        remaining_questions = [qa for qa in questions if qa["question"] not in skip_questions]
        
        if not remaining_questions:
            logger.info(f"  All {len(questions)} questions already completed, skipping")
            return []
        
        if skip_questions:
            logger.info(f"  Skipping {len(skip_questions)} completed, {len(remaining_questions)} remaining")
        
        # Check document size and decide routing
        doc_tokens = estimate_doc_tokens(doc)
        use_rag = doc_tokens >= TOKEN_THRESHOLD
        
        if use_rag:
            logger.info(f"  Large doc ({doc_tokens:,} tokens) - using RAG mode")
            try:
                chunked_doc = self.chunker.chunk_document(doc)
                retriever = ChunkRetriever(chunked_doc)
                logger.info(f"  Created {len(chunked_doc.chunks)} chunks")
            except Exception as e:
                logger.error(f"  Failed to chunk document: {e}")
                return []
        else:
            logger.info(f"  Small doc ({doc_tokens:,} tokens) - using full doc mode")
            doc_content = self.get_document_content(folder, data_dir)
            retriever = None
        
        results = []
        for i, qa in enumerate(remaining_questions):
            question = qa["question"]
            logger.info(f"  [Folder {folder}] Q{i+1}/{len(remaining_questions)}: {question[:50]}...")
            
            try:
                if use_rag and retriever:
                    # RAG mode: retrieve relevant chunks + page images
                    chunks, pages = retriever.retrieve(question)
                    rag_content = build_rag_content(chunks, doc, pages, self.max_images, self.include_images)
                    answer, input_tokens, output_tokens = self.call_mistral(
                        rag_content, question, folder=folder
                    )
                else:
                    # Full doc mode
                    answer, input_tokens, output_tokens = self.call_mistral(
                        doc_content, question, folder=folder
                    )
                self._update_tokens(input_tokens, output_tokens)
            except Exception as e:
                logger.error(f"  Error on folder {folder} Q{i+1}: {e}")
                answer = ""
            
            result = {
                "question": question,
                "answer": qa.get("answer", ""),
                "evidence": qa.get("evidence", ""),
                "type": qa.get("type", ""),
                "sys_ans": answer,
                "file": str(folder),
                "model": self.model,
                "mode": "rag" if use_rag else "full"
            }
            results.append(result)
        
        return results
    
    def load_completed(self, output_file: Path) -> dict[str, set[str]]:
        """
        Load already completed question+folder pairs from results file.
        
        Returns:
            Dict mapping folder -> set of completed questions (with non-empty answers)
        """
        completed = {}
        
        if not output_file.exists():
            return completed
        
        with open(output_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    result = json.loads(line)
                    folder = str(result.get("file", ""))
                    question = result.get("question", "")
                    answer = result.get("sys_ans", "")
                    
                    # Only count as completed if answer is non-empty
                    if folder and question and answer:
                        if folder not in completed:
                            completed[folder] = set()
                        completed[folder].add(question)
                except json.JSONDecodeError:
                    continue
        
        return completed
    
    def run(self, data_dir: str, start: int, end: int, output_dir: str = None, resume: bool = True, domain: str = None):
        """Run benchmark on all folders from start to end.
        
        Args:
            data_dir: Directory containing document folders
            start: Starting folder number
            end: Ending folder number
            output_dir: Output directory (default: same as data_dir)
            resume: If True, skip already completed questions (default: True)
            domain: Filter by domain (academic, finance, government, legal, news)
        """
        if output_dir is None:
            output_dir = data_dir
        
        # If running a domain slice, write to a domain-specific results file so
        # finance/legal/etc. runs don't mix into one giant JSONL.
        output_suffix = f"_{domain}" if domain else ""
        output_file = Path(output_dir) / f"{self.model_short}{output_suffix}_results.jsonl"
        
        # Filter by domain if specified
        if domain:
            domain_folders = get_domain_folders(domain)
            folders = [f for f in range(start, end + 1) if f in domain_folders]
            logger.info(f"Filtering by domain '{domain}': {len(folders)} folders")
        else:
            folders = list(range(start, end + 1))
        
        if not folders:
            logger.warning(f"No folders to process (domain filter may have excluded all)")
            return []
        
        # Load existing progress
        completed = {}
        if resume and output_file.exists():
            completed = self.load_completed(output_file)
            total_completed = sum(len(qs) for qs in completed.values())
            logger.info(f"Resuming: found {total_completed} completed questions in {len(completed)} folders")
        
        max_folder = max(folders) if folders else end
        logger.info(f"Running {self.model} on {len(folders)} folders (range {min(folders)}-{max_folder}) with {self.workers} worker(s)")
        logger.info(f"Output: {output_file}")
        all_results = []
        file_lock = threading.Lock()
        
        def process_folder(folder: int) -> list[dict]:
            folder_str = str(folder)
            completed_qs = completed.get(folder_str, set())
            
            logger.info(f"Processing folder {folder}/{max_folder}")
            results = self.run_folder(folder, data_dir, skip_questions=completed_qs)
            
            # Thread-safe file write
            with file_lock:
                with open(output_file, 'a') as f:
                    for result in results:
                        f.write(json.dumps(result) + '\n')
            
            return results
        
        if self.workers == 1:
            # Sequential processing
            for folder in folders:
                results = process_folder(folder)
                all_results.extend(results)
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = {executor.submit(process_folder, folder): folder for folder in folders}
                
                for future in as_completed(futures):
                    folder = futures[future]
                    try:
                        results = future.result()
                        all_results.extend(results)
                    except Exception as e:
                        logger.error(f"Folder {folder} failed: {e}")
        
        self.print_cost_summary()
        
        return all_results
    
    def print_cost_summary(self):
        """Print token usage and estimated cost."""
        costs = MODEL_COSTS.get(self.model, {"input": 0, "output": 0})
        input_cost = (self.total_input_tokens / 1_000_000) * costs["input"]
        output_cost = (self.total_output_tokens / 1_000_000) * costs["output"]
        total_cost = input_cost + output_cost
        
        logger.info("=" * 50)
        logger.info(f"Token Usage Summary for {self.model}:")
        logger.info(f"  Input tokens:  {self.total_input_tokens:,}")
        logger.info(f"  Output tokens: {self.total_output_tokens:,}")
        logger.info(f"  Estimated cost: ${total_cost:.4f}")
        logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Run Mistral models on DocBench")
    parser.add_argument(
        "--model", 
        type=str, 
        default="ministral-8b-latest",
        choices=list(MISTRAL_MODELS.keys()),
        help="Mistral model to use"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory containing document folders (default: ./data)"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting folder number (default: 0)"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=228,
        help="Ending folder number (default: 228)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as data-dir)"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=8,
        help="Maximum images per document (default: 8, API limit)"
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=200,
        help="Minimum image dimension in pixels to include (default: 200, filters out icons)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, don't resume from previous progress"
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=list(DOMAINS.keys()),
        help="Filter by domain: academic (49 docs), finance (40 docs), government (44 docs), legal (46 docs), news (50 docs)"
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Disable image extraction and inclusion (text-only mode, faster and avoids rate limits)"
    )
    
    args = parser.parse_args()
    
    runner = MistralRunner(
        model_name=args.model,
        max_images=args.max_images,
        min_image_size=args.min_image_size,
        workers=args.workers,
        include_images=not args.no_images
    )
    runner.run(
        data_dir=args.data_dir,
        start=args.start,
        end=args.end,
        output_dir=args.output_dir,
        resume=not args.no_resume,
        domain=args.domain
    )


if __name__ == "__main__":
    main()