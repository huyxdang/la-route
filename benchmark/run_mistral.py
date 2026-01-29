"""
Mistral API runner for DocBench benchmark.

Runs Mistral models on DocBench PDFs with full multimodal support (text + images).

Usage:
    python run_mistral.py --model ministral-8b-latest --start 0 --end 228
    python run_mistral.py --model mistral-large-latest --start 0 --end 228 --workers 4
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


class MistralRunner:
    """Runs Mistral models on DocBench documents with multimodal support."""
    
    def __init__(self, model_name: str, max_images: int = 10, workers: int = 1):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment")
        
        self.client = Mistral(api_key=api_key)
        self.model = MISTRAL_MODELS.get(model_name, model_name)
        self.model_short = model_name.replace("-latest", "")
        self.max_images = max_images
        self.workers = workers
        
        # Track costs (thread-safe)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._lock = threading.Lock()
        
        logger.info(f"Initialized MistralRunner with model: {self.model}, workers: {workers}")
    
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
        doc = extract_pdf(str(pdf_path))
        
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
        
        # Add images with page annotations
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
    def call_mistral(self, doc_content: list[dict], question: str) -> tuple[str, int, int]:
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
        
        response = self.client.chat.complete(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            max_tokens=1000,
            temperature=0.0
        )
        
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
    
    def run_folder(self, folder: str, data_dir: str) -> list[dict]:
        """Run all questions for a single document folder."""
        folder_path = Path(data_dir) / str(folder)
        
        if not folder_path.exists():
            logger.warning(f"Folder {folder} does not exist, skipping")
            return []
        
        try:
            questions = self.get_questions(folder, data_dir)
            doc_content = self.get_document_content(folder, data_dir)
        except FileNotFoundError as e:
            logger.error(f"Skipping folder {folder}: {e}")
            return []
        
        results = []
        for i, qa in enumerate(questions):
            question = qa["question"]
            logger.info(f"  [Folder {folder}] Q{i+1}/{len(questions)}: {question[:50]}...")
            
            try:
                answer, input_tokens, output_tokens = self.call_mistral(doc_content, question)
                self._update_tokens(input_tokens, output_tokens)
            except Exception as e:
                logger.error(f"  Error on Q{i+1}: {e}")
                answer = ""
            
            result = {
                "question": question,
                "answer": qa.get("answer", ""),
                "evidence": qa.get("evidence", ""),
                "type": qa.get("type", ""),
                "sys_ans": answer,
                "file": str(folder),
                "model": self.model
            }
            results.append(result)
        
        return results
    
    def run(self, data_dir: str, start: int, end: int, output_dir: str = None):
        """Run benchmark on all folders from start to end."""
        if output_dir is None:
            output_dir = data_dir
        
        output_file = Path(output_dir) / f"{self.model_short}_results.jsonl"
        
        logger.info(f"Running {self.model} on folders {start}-{end} with {self.workers} worker(s)")
        logger.info(f"Output: {output_file}")
        
        folders = list(range(start, end + 1))
        all_results = []
        file_lock = threading.Lock()
        
        def process_folder(folder: int) -> list[dict]:
            logger.info(f"Processing folder {folder}/{end}")
            results = self.run_folder(folder, data_dir)
            
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
        default=10,
        help="Maximum images per document (default: 10)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)"
    )
    
    args = parser.parse_args()
    
    runner = MistralRunner(
        model_name=args.model,
        max_images=args.max_images,
        workers=args.workers
    )
    runner.run(
        data_dir=args.data_dir,
        start=args.start,
        end=args.end,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
