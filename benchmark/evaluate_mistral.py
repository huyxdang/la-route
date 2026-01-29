"""
Evaluate Mistral model outputs on DocBench.

This uses Mistral Large as the evaluator (instead of GPT-4) to judge
whether answers are correct.

Usage:
    python evaluate_mistral.py --input ministral-8b_results.jsonl
    python evaluate_mistral.py --input mistral-large_results.jsonl
"""

import json
import argparse
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from mistralai import Mistral
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


EVAL_PROMPT = """Task Overview:
You are tasked with evaluating user answers based on a given question, reference answer, and additional reference text. Your goal is to assess the correctness of the user answer using a specific metric.

Evaluation Criteria:
1. Yes/No Questions: Verify if the user's answer aligns with the reference answer in terms of a "yes" or "no" response.
2. Short Answers/Directives: Ensure key details such as numbers, specific nouns/verbs, and dates match those in the reference answer.
3. Abstractive/Long Answers: The user's answer can differ in wording but must convey the same meaning and contain the same key information as the reference answer to be considered correct.

Evaluation Process:
1. Identify the type of question presented.
2. Apply the relevant criteria from the Evaluation Criteria.
3. Compare the user's answer against the reference answer accordingly.
4. Consult the reference text for clarification when needed.
5. Score the answer with a binary label 0 or 1, where 0 denotes wrong and 1 denotes correct.
NOTE that if the user answer is 0 or an empty string, it should get a 0 score.

Question: {question}
User Answer: {sys_ans}
Reference Answer: {ref_ans}
Reference Text: {ref_text}

Evaluation Form (respond with ONLY the number 0 or 1):
Correctness: """


class MistralEvaluator:
    """Evaluates DocBench answers using Mistral Large."""
    
    def __init__(self, evaluator_model: str = "mistral-large-latest"):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment")
        
        self.client = Mistral(api_key=api_key)
        self.model = evaluator_model
        self.total_tokens = 0
    
    @retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def evaluate_single(self, question: str, sys_ans: str, ref_ans: str, ref_text: str) -> int:
        """Evaluate a single answer. Returns 0 or 1."""
        prompt = EVAL_PROMPT.format(
            question=question,
            sys_ans=sys_ans,
            ref_ans=ref_ans,
            ref_text=ref_text
        )
        
        response = self.client.chat.complete(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful evaluator. Respond with only 0 or 1."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=5,
            temperature=0.0
        )
        
        if hasattr(response, 'usage') and response.usage:
            self.total_tokens += response.usage.prompt_tokens + response.usage.completion_tokens
        
        answer = response.choices[0].message.content.strip()
        
        # Parse the score
        if "1" in answer:
            return 1
        else:
            return 0
    
    def evaluate_file(self, input_path: str, output_path: str = None, resume_id: int = 0):
        """Evaluate all answers in a results file."""
        input_path = Path(input_path)
        
        if output_path is None:
            output_path = input_path.with_suffix('.eval.jsonl')
        else:
            output_path = Path(output_path)
        
        # Load results
        with open(input_path, 'r') as f:
            results = [json.loads(line) for line in f if line.strip()]
        
        logger.info(f"Evaluating {len(results)} answers from {input_path}")
        
        scores = []
        
        for i, result in enumerate(results):
            if i < resume_id:
                continue
            
            question = result["question"]
            sys_ans = result["sys_ans"]
            ref_ans = result.get("answer", "")
            ref_text = result.get("evidence", "")
            
            try:
                score = self.evaluate_single(question, sys_ans, ref_ans, ref_text)
            except Exception as e:
                logger.error(f"Error evaluating Q{i}: {e}")
                score = 0
            
            result["eval_score"] = score
            scores.append(score)
            
            # Write incrementally
            with open(output_path, 'a') as f:
                f.write(json.dumps(result) + '\n')
            
            if (i + 1) % 50 == 0:
                acc = sum(scores) / len(scores) * 100
                logger.info(f"Progress: {i+1}/{len(results)}, Running Accuracy: {acc:.1f}%")
        
        # Final summary
        accuracy = sum(scores) / len(scores) * 100 if scores else 0
        logger.info("=" * 50)
        logger.info(f"Final Results:")
        logger.info(f"  Total questions: {len(scores)}")
        logger.info(f"  Correct: {sum(scores)}")
        logger.info(f"  Accuracy: {accuracy:.2f}%")
        logger.info(f"  Eval tokens used: {self.total_tokens:,}")
        logger.info("=" * 50)
        
        return accuracy, scores


def compare_results(file1: str, file2: str):
    """Compare results from two models to identify routing opportunities."""
    with open(file1, 'r') as f:
        results1 = [json.loads(line) for line in f if line.strip()]
    with open(file2, 'r') as f:
        results2 = [json.loads(line) for line in f if line.strip()]
    
    if len(results1) != len(results2):
        logger.warning(f"Result counts differ: {len(results1)} vs {len(results2)}")
    
    # Ensure we have eval scores
    if "eval_score" not in results1[0] or "eval_score" not in results2[0]:
        logger.error("Run evaluation first to get eval_score")
        return
    
    both_correct = 0
    small_only = 0  # Small correct, large wrong
    large_only = 0  # Large correct, small wrong
    both_wrong = 0
    
    for r1, r2 in zip(results1, results2):
        s1 = r1.get("eval_score", 0)
        s2 = r2.get("eval_score", 0)
        
        if s1 and s2:
            both_correct += 1
        elif s1 and not s2:
            small_only += 1
        elif not s1 and s2:
            large_only += 1
        else:
            both_wrong += 1
    
    total = len(results1)
    
    logger.info("=" * 50)
    logger.info("Model Comparison Analysis:")
    logger.info(f"  Both correct (use small):     {both_correct} ({both_correct/total*100:.1f}%)")
    logger.info(f"  Only small correct:           {small_only} ({small_only/total*100:.1f}%)")
    logger.info(f"  Only large correct (need lg): {large_only} ({large_only/total*100:.1f}%)")
    logger.info(f"  Both wrong:                   {both_wrong} ({both_wrong/total*100:.1f}%)")
    logger.info("-" * 50)
    logger.info(f"  Routing potential: {both_correct/total*100:.1f}% can use cheap model")
    logger.info(f"  Must use large:    {large_only/total*100:.1f}% need expensive model")
    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Mistral DocBench results")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file with model results"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL file with eval scores (default: input.eval.jsonl)"
    )
    parser.add_argument(
        "--resume",
        type=int,
        default=0,
        help="Resume from this question ID"
    )
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help="Second results file to compare (for routing analysis)"
    )
    parser.add_argument(
        "--evaluator",
        type=str,
        default="mistral-large-latest",
        help="Model to use for evaluation (default: mistral-large-latest)"
    )
    
    args = parser.parse_args()
    
    if args.compare:
        compare_results(args.input, args.compare)
    else:
        evaluator = MistralEvaluator(args.evaluator)
        evaluator.evaluate_file(args.input, args.output, args.resume)


if __name__ == "__main__":
    main()
