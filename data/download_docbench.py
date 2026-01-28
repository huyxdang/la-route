"""
Download DocBench dataset from HuggingFace.
DocBench is a benchmark for document-grounded question answering.
"""

import os
import json
from pathlib import Path
from tqdm import tqdm

# Try multiple potential sources for DocBench or similar datasets
DOCBENCH_SOURCES = [
    "Anni-Zou/DocBench",  # Primary source from context
    "docbench/docbench",   # Alternative
]

# Fallback: Create synthetic data for demo purposes
SYNTHETIC_DATA_SIZE = 200


def download_docbench(output_dir: str = "data/raw/docbench"):
    """
    Download DocBench dataset from HuggingFace.
    Falls back to synthetic data if DocBench is not available.
    """
    from datasets import load_dataset
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Try to load DocBench
    dataset = None
    for source in DOCBENCH_SOURCES:
        try:
            print(f"Attempting to load dataset from: {source}")
            dataset = load_dataset(source)
            print(f"✓ Successfully loaded from {source}")
            break
        except Exception as e:
            print(f"  Could not load from {source}: {e}")
    
    if dataset is not None:
        # Save the dataset
        for split in dataset.keys():
            split_path = output_path / f"{split}.jsonl"
            print(f"Saving {split} split to {split_path}")
            
            with open(split_path, "w") as f:
                for item in tqdm(dataset[split], desc=f"Saving {split}"):
                    f.write(json.dumps(dict(item)) + "\n")
            
            print(f"  Saved {len(dataset[split])} examples")
        
        # Print statistics
        print_dataset_stats(dataset)
        return output_path
    
    else:
        print("\n⚠ DocBench not available. Generating synthetic data for demo...")
        return generate_synthetic_data(output_path)


def generate_synthetic_data(output_path: Path):
    """
    Generate synthetic document Q&A data for demo and testing.
    This simulates the DocBench format.
    """
    import random
    
    # Sample document templates
    DOCUMENT_TEMPLATES = [
        {
            "type": "policy",
            "title": "Employee Handbook - {company}",
            "sections": [
                "Section 1: Introduction\nThis handbook outlines the policies and procedures for all employees at {company}. All employees must familiarize themselves with these guidelines.",
                "Section 2: Working Hours\nStandard working hours are 9:00 AM to 5:00 PM, Monday through Friday. Flexible arrangements require manager approval.",
                "Section 3: Leave Policy\nEmployees are entitled to {leave_days} days of paid leave per year. Leave must be requested at least 2 weeks in advance.",
                "Section 4: Expense Reimbursement\nBusiness expenses must be submitted within {expense_days} days. Receipts are required for all expenses over $25.",
                "Section 5: Code of Conduct\nEmployees must maintain professional behavior at all times. Violations may result in disciplinary action up to and including termination.",
            ]
        },
        {
            "type": "contract",
            "title": "Service Agreement - {company}",
            "sections": [
                "Article 1: Parties\nThis agreement is entered into between {company} ('Provider') and the Client ('Recipient').",
                "Article 2: Services\nThe Provider agrees to deliver {service_type} services as outlined in Exhibit A.",
                "Article 3: Payment Terms\nPayment is due within {payment_days} days of invoice. Late payments incur a {late_fee}% monthly fee.",
                "Article 4: Term and Termination\nThis agreement shall remain in effect for {contract_years} year(s). Either party may terminate with {notice_days} days written notice.",
                "Article 5: Confidentiality\nBoth parties agree to maintain confidentiality of all proprietary information exchanged during the term of this agreement.",
            ]
        },
        {
            "type": "technical",
            "title": "API Documentation - {product}",
            "sections": [
                "Overview\nThe {product} API provides programmatic access to our platform. All requests require authentication via API key.",
                "Authentication\nInclude your API key in the X-API-Key header. Rate limits: {rate_limit} requests per minute for standard tier.",
                "Endpoints\nPOST /api/v1/query - Submit a query\nGET /api/v1/results/[id] - Retrieve results\nDELETE /api/v1/session/[id] - Delete session",
                "Error Codes\n400: Bad Request\n401: Unauthorized\n429: Rate Limited\n500: Internal Server Error",
                "Examples\ncurl -X POST https://api.example.com/v1/query -H 'X-API-Key: YOUR_KEY' -d 'query=example'",
            ]
        },
        {
            "type": "legal",
            "title": "Terms of Service - {company}",
            "sections": [
                "1. Acceptance of Terms\nBy accessing or using {company}'s services, you agree to be bound by these Terms of Service.",
                "2. User Responsibilities\nUsers must not engage in any activity that violates applicable laws or infringes on the rights of others.",
                "3. Intellectual Property\nAll content and materials available through {company} are the property of {company} or its licensors.",
                "4. Limitation of Liability\n{company} shall not be liable for any indirect, incidental, special, or consequential damages.",
                "5. Dispute Resolution\nAny disputes arising from these terms shall be resolved through binding arbitration in {jurisdiction}.",
            ]
        },
    ]
    
    # Question templates by difficulty
    SIMPLE_QUESTIONS = [
        ("What are the working hours?", "working hours"),
        ("How many leave days are provided?", "leave days"),
        ("What is the expense submission deadline?", "expense"),
        ("What is the payment deadline?", "payment"),
        ("What is the rate limit?", "rate limit"),
        ("How do I authenticate?", "authentication"),
        ("What is the contract term?", "term"),
        ("How much notice is required for termination?", "notice"),
    ]
    
    COMPLEX_QUESTIONS = [
        ("What are the consequences of violating the code of conduct?", "conduct"),
        ("Compare the termination clauses with the notice requirements", "termination"),
        ("Explain the relationship between late payments and contract termination", "payment"),
        ("What happens if both parties breach confidentiality?", "confidentiality"),
        ("How does the liability limitation affect dispute resolution?", "liability"),
        ("What are all the error scenarios and how should they be handled?", "error"),
    ]
    
    # Generate data
    data = []
    companies = ["Acme Corp", "TechFlow Inc", "GlobalServ", "InnovateCo", "DataPrime"]
    
    for i in tqdm(range(SYNTHETIC_DATA_SIZE), desc="Generating synthetic data"):
        template = random.choice(DOCUMENT_TEMPLATES)
        company = random.choice(companies)
        
        # Fill in template variables
        doc_text = "\n\n".join(template["sections"])
        doc_text = doc_text.format(
            company=company,
            leave_days=random.choice([15, 20, 25]),
            expense_days=random.choice([30, 45, 60]),
            payment_days=random.choice([15, 30, 45]),
            late_fee=random.choice([1.5, 2, 2.5]),
            contract_years=random.choice([1, 2, 3]),
            notice_days=random.choice([30, 60, 90]),
            service_type=random.choice(["consulting", "software development", "data analytics"]),
            product=random.choice(["DataHub", "QueryEngine", "InsightAPI"]),
            rate_limit=random.choice([60, 100, 1000]),
            jurisdiction=random.choice(["Delaware", "California", "New York"]),
        )
        
        title = template["title"].format(company=company, product="DataHub")
        
        # Generate question-answer pairs
        is_complex = random.random() > 0.7  # 30% complex
        if is_complex:
            q_template, keyword = random.choice(COMPLEX_QUESTIONS)
        else:
            q_template, keyword = random.choice(SIMPLE_QUESTIONS)
        
        # Find relevant section
        relevant_section = ""
        for section in template["sections"]:
            if keyword.lower() in section.lower():
                relevant_section = section
                break
        
        if not relevant_section:
            relevant_section = template["sections"][0]
        
        # Generate answer from relevant section
        answer = relevant_section.split('\n')[1] if '\n' in relevant_section else relevant_section[:200]
        
        data.append({
            "id": f"synthetic_{i:04d}",
            "document": f"{title}\n\n{doc_text}",
            "question": q_template,
            "answer": answer,
            "doc_type": template["type"],
            "is_complex": is_complex,
            "relevant_section": relevant_section,
        })
    
    # Split into train/val
    random.shuffle(data)
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    # Save
    for split_name, split_data in [("train", train_data), ("validation", val_data)]:
        split_path = output_path / f"{split_name}.jsonl"
        with open(split_path, "w") as f:
            for item in split_data:
                f.write(json.dumps(item) + "\n")
        print(f"✓ Saved {len(split_data)} examples to {split_path}")
    
    # Print statistics
    print("\n=== Synthetic Dataset Statistics ===")
    print(f"Total examples: {len(data)}")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"Complex questions: {sum(1 for d in data if d['is_complex'])} ({sum(1 for d in data if d['is_complex'])/len(data)*100:.1f}%)")
    
    doc_types = {}
    for d in data:
        doc_types[d['doc_type']] = doc_types.get(d['doc_type'], 0) + 1
    print(f"Doc types: {doc_types}")
    
    return output_path


def print_dataset_stats(dataset):
    """Print statistics about the loaded dataset."""
    print("\n=== DocBench Dataset Statistics ===")
    
    for split in dataset.keys():
        print(f"\n{split} split:")
        print(f"  Examples: {len(dataset[split])}")
        
        if len(dataset[split]) > 0:
            sample = dataset[split][0]
            print(f"  Fields: {list(sample.keys())}")
            
            # Document length stats
            if 'document' in sample:
                doc_lengths = [len(d['document']) for d in dataset[split]]
                print(f"  Avg doc length: {sum(doc_lengths)/len(doc_lengths):.0f} chars")
            
            # Question length stats
            if 'question' in sample:
                q_lengths = [len(d['question']) for d in dataset[split]]
                print(f"  Avg question length: {sum(q_lengths)/len(q_lengths):.0f} chars")


if __name__ == "__main__":
    download_docbench()
