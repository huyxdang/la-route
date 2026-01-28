"""
System prompts for Le-Route grounded Q&A.
"""

SYSTEM_PROMPT = """You are a precise document Q&A assistant. Your role is to answer questions based ONLY on the provided document chunks.

## Rules:
1. ONLY use information from the provided chunks to answer
2. ALWAYS cite your sources using [1], [2], etc. corresponding to chunk numbers
3. If the answer is not in the provided chunks, say "I cannot find this information in the document."
4. Be concise and direct in your answers
5. If multiple chunks support an answer, cite all relevant ones
6. Never make up information or use external knowledge

## Citation Format:
- Place citations immediately after the relevant statement
- Example: "The deadline is March 31st [1]."
- Multiple citations: "Employees must follow the code of conduct [1][2]."

## Abstention:
If the question cannot be answered from the provided chunks:
- Say: "I cannot find this information in the document."
- Do NOT guess or use external knowledge
- Do NOT apologize excessively"""


def build_qa_prompt(question: str, chunks: list[tuple[str, float, int]]) -> str:
    """
    Build the user prompt with question and relevant chunks.
    
    Args:
        question: User's question
        chunks: List of (chunk_text, relevance_score, chunk_id) tuples
    
    Returns:
        Formatted prompt string
    """
    chunks_text = ""
    for chunk_text, score, chunk_id in chunks:
        chunks_text += f"\n[{chunk_id + 1}] (relevance: {score:.2f})\n{chunk_text}\n"
    
    return f"""## Document Chunks:
{chunks_text}

## Question:
{question}

## Instructions:
Answer the question using ONLY the information from the chunks above. Cite sources using [1], [2], etc.
If the information is not in the chunks, say "I cannot find this information in the document."

## Answer:"""


def build_highlight_prompt(answer: str, chunk_text: str) -> str:
    """
    Build a prompt to identify which part of the chunk was used for the answer.
    Used for highlighting in the citation panel.
    
    Args:
        answer: The generated answer
        chunk_text: The full chunk text
    
    Returns:
        Prompt for highlight extraction
    """
    return f"""Given this answer and the source chunk, identify the exact substring from the chunk that supports the answer.

Answer: {answer}

Chunk text:
{chunk_text}

Return ONLY the exact substring from the chunk (verbatim, no modifications). If the entire chunk is relevant, return the most important sentence."""


# Risk keywords for routing decisions
HIGH_RISK_KEYWORDS = [
    "terminate", "termination", "fire", "fired", "lawsuit", "sue", "legal action",
    "liability", "damages", "penalty", "penalties", "violation", "breach",
    "discrimination", "harassment", "wrongful", "negligence", "fraud",
    "confidential", "proprietary", "trade secret", "non-compete"
]

MEDIUM_RISK_KEYWORDS = [
    "compliance", "regulation", "regulatory", "requirement", "obligation",
    "penalty", "fine", "audit", "approval required",
    "authorization", "permission required", "consent required", 
    "binding agreement", "contractual"
]

COMPLEX_QUERY_INDICATORS = [
    "compare", "contrast", "difference", "versus", "vs",
    "relationship", "impact", "effect", "consequence",
    "why", "how does", "explain", "analyze", "evaluate",
    "multiple", "several", "all", "every", "any"
]


def detect_risk_level(query: str) -> str:
    """
    Detect the risk level of a query based on keywords.
    
    Returns:
        "high", "medium", or "low"
    """
    query_lower = query.lower()
    
    for keyword in HIGH_RISK_KEYWORDS:
        if keyword in query_lower:
            return "high"
    
    for keyword in MEDIUM_RISK_KEYWORDS:
        if keyword in query_lower:
            return "medium"
    
    return "low"


def is_complex_query(query: str) -> bool:
    """
    Detect if a query is complex based on indicators.
    """
    query_lower = query.lower()
    
    # Check for complex query indicators
    for indicator in COMPLEX_QUERY_INDICATORS:
        if indicator in query_lower:
            return True
    
    # Long queries are often complex
    if len(query.split()) > 20:
        return True
    
    # Multiple question marks suggest complexity
    if query.count("?") > 1:
        return True
    
    return False
