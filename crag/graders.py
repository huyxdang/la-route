"""
LLM Graders with Structured Output for CRAG workflow.

Uses Pydantic models with Mistral models for structured, reliable decisions:
1. Query Router (ministral-3b) - fast routing to vectorstore or web search
2. Query Rewriter (mistral-small) - rewrites queries with conversation history context
3. Document Grader (mistral-large) - grades document relevance with full context
4. Document Reranker (Cohere) - reranks retrieved docs by relevance
5. Generation Grader (mistral-small) - combined hallucination + answer check
"""

import os
import json
from typing import Literal, List, Optional, Any
from pydantic import BaseModel, Field

# Model assignments per grader (optimized for cost/quality tradeoff)
ROUTER_MODEL = "ministral-3b-latest"        # Fast, cheap routing
REWRITER_MODEL = "mistral-small-latest"     # Query rewriting with history
DOC_GRADER_MODEL = "mistral-large-latest"   # Best context understanding
GEN_GRADER_MODEL = "mistral-small-latest"   # Good balance for grading
DEFAULT_MODEL = "mistral-large-latest"       # Fallback / generation

MAX_DOC_LENGTH = 2000      # For grading
MAX_RERANK_LENGTH = 4000   # For Cohere
MAX_HISTORY_LENGTH = 500   # For query rewriting

# ============== Pydantic Models for Structured Output ==============

class RouteQuery(BaseModel):
    """Router decision for query classification."""
    datasource: Literal["vectorstore", "web_search"] = Field(
        description="Route to 'vectorstore' for technical papers/indexed content, "
                    "'web_search' for general knowledge or current events."
    )
    reasoning: str = Field(
        description="Brief explanation for the routing decision."
    )


class GradeDocumentsBatch(BaseModel):
    """Batch relevance scores for multiple documents."""
    scores: List[Literal["yes", "no"]] = Field(
        description="List of relevance scores ('yes' or 'no'), one per document in order."
    )


class GradeGeneration(BaseModel):
    """Combined generation quality check - single LLM call for both hallucination and answer quality."""
    is_grounded: Literal["yes", "no"] = Field(
        description="Is the answer grounded in the documents? 'yes' if all claims are supported, 'no' if hallucinated."
    )
    answers_question: Literal["yes", "no"] = Field(
        description="Does the answer address the question? 'yes' or 'no'."
    )
    reasoning: str = Field(
        description="Brief explanation of the grading decision."
    )


class RewrittenQuery(BaseModel):
    """Rewritten standalone query from conversation context."""
    query: str = Field(
        description="A clean, standalone search query that captures the user's intent without requiring conversation history."
    )
    reasoning: str = Field(
        description="Brief explanation of how the query was rewritten."
    )


# ============== Mistral Client with Structured Output ==============

class MistralStructuredClient:
    """
    Mistral client wrapper that supports structured output via JSON mode.
    
    Uses Mistral's JSON mode with Pydantic schema for reliable structured responses.
    """
    
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self._client = None
    
    def _get_client(self):
        """Lazy init Mistral client."""
        if self._client is None:
            from mistralai import Mistral
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("MISTRAL_API_KEY not found in environment")
            self._client = Mistral(api_key=api_key)
        return self._client
    
    def invoke_structured(
        self, 
        messages: List[dict], 
        response_model: type[BaseModel]
    ) -> BaseModel:
        """
        Invoke Mistral with structured JSON output.
        
        Args:
            messages: List of message dicts with role/content
            response_model: Pydantic model for the response
            
        Returns:
            Parsed Pydantic model instance
        """
        client = self._get_client()
        
        # Build example JSON showing expected fields (not the full schema)
        schema = response_model.model_json_schema()
        properties = schema.get("properties", {})
        
        # Create a simple example with field names and types
        example_fields = []
        for field_name, field_info in properties.items():
            field_type = field_info.get("type", "string")
            # Handle Literal/enum types
            if "enum" in field_info:
                example_value = field_info["enum"][0]
            elif "anyOf" in field_info:
                # Handle Optional or Union types
                for option in field_info["anyOf"]:
                    if "enum" in option:
                        example_value = option["enum"][0]
                        break
                else:
                    example_value = "..."
            elif field_type == "array":
                example_value = ["yes", "no"]
            elif field_type == "string":
                example_value = "..."
            else:
                example_value = "..."
            example_fields.append(f'  "{field_name}": "{example_value}"' if isinstance(example_value, str) else f'  "{field_name}": {json.dumps(example_value)}')
        
        example_json = "{\n" + ",\n".join(example_fields) + "\n}"
        
        # Build field descriptions
        field_descriptions = []
        for field_name, field_info in properties.items():
            desc = field_info.get("description", "")
            if "enum" in field_info:
                desc += f" (allowed values: {field_info['enum']})"
            field_descriptions.append(f"- {field_name}: {desc}")
        
        fields_text = "\n".join(field_descriptions)
        
        # Prepend schema instruction to system message
        enhanced_messages = messages.copy()
        if enhanced_messages and enhanced_messages[0]["role"] == "system":
            enhanced_messages[0]["content"] += f"""

Respond with a JSON object containing these fields:
{fields_text}

Example format:
{example_json}

Respond ONLY with a valid JSON object, no other text."""
        
        response = client.chat.complete(
            model=self.model,
            messages=enhanced_messages,
            response_format={"type": "json_object"},
            temperature=0
        )
        
        content = response.choices[0].message.content
        
        # Parse JSON and validate with Pydantic
        try:
            data = json.loads(content)
            return response_model.model_validate(data)
        except (json.JSONDecodeError, Exception) as e:
            # Fallback: try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return response_model.model_validate(data)
            raise ValueError(f"Failed to parse structured response: {e}\nContent: {content}")


# ============== Grader Classes ==============

class QueryRouter:
    """
    Routes queries to appropriate datasource.
    
    Uses ministral-3b for fast, cheap routing decisions.
    
    - Technical papers, indexed content → vectorstore
    - General knowledge, current events → web_search
    """
    
    SYSTEM_PROMPT = """You are an expert at routing user questions to the appropriate data source.

You have access to two data sources:

1. **vectorstore**: Contains ONLY NeurIPS 2025 accepted papers (6,000+ papers). 
   Topics: machine learning, deep learning, optimization, reinforcement learning, NLP, computer vision, AI theory.
   Use this for: questions about NeurIPS 2025 research, methods, architectures, benchmarks, or technical ML concepts.
   
2. **web_search**: For everything else — other conferences (ICML, ICLR, ACL), other years, 
   current events, company news, general knowledge, or topics outside academic ML.

Important: The vectorstore contains ONLY NeurIPS 2025. Questions about other venues, years, or non-research topics should use web_search.

Based on the question, decide which source is most appropriate."""

    def __init__(self, model: str = ROUTER_MODEL):
        self.model = model
        self._client = None
        
    def _get_client(self) -> MistralStructuredClient:
        """Lazy init Mistral client."""
        if self._client is None:
            self._client = MistralStructuredClient(self.model)
        return self._client
    
    def route(self, question: str) -> RouteQuery:
        """
        Route a question to the appropriate datasource.
        
        Args:
            question: The user's question
            
        Returns:
            RouteQuery with datasource and reasoning
        """
        client = self._get_client()
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}"}
        ]
        
        return client.invoke_structured(messages, RouteQuery)


class DocumentGrader:
    """
    Grades document relevance to a question using batch mode.
    
    Uses mistral-large for best context understanding when grading documents.
    Uses a single LLM call to grade all documents at once instead of N calls.
    """
    
    SYSTEM_PROMPT = """You are a grader assessing the relevance of retrieved documents to a user question.

Your task:
1. For each document, check if it contains keywords or semantic meaning relevant to the question
2. Documents do NOT need to fully answer the question - just be relevant/useful
3. Return a list of 'yes' or 'no' scores, one per document in order

Be lenient - if there's any reasonable connection, mark it as relevant."""

    def __init__(self, model: str = DOC_GRADER_MODEL):
        self.model = model
        self._client = None
        
    def _get_client(self) -> MistralStructuredClient:
        if self._client is None:
            self._client = MistralStructuredClient(self.model)
        return self._client
    
    def grade_documents(
        self, 
        documents: List, 
        question: str
    ) -> tuple[List, bool]:
        """
        Grade multiple documents in a single LLM call and filter irrelevant ones.
        
        Args:
            documents: List of Document objects
            question: The user's question
            
        Returns:
            Tuple of (filtered_documents, needs_web_search)
        """
        if not documents:
            return [], True
        
        client = self._get_client()
        
        # Build batch prompt with numbered documents
        doc_parts = []
        for i, doc in enumerate(documents):
            content = doc.content if hasattr(doc, 'content') else str(doc)
            # Truncate long documents to avoid token limits
            content = content[:MAX_DOC_LENGTH] if len(content) > MAX_DOC_LENGTH else content
            doc_parts.append(f"[Document {i+1}]\n{content}")
        
        documents_text = "\n\n---\n\n".join(doc_parts)
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"""Documents:
{documents_text}

Question: {question}

For each document (1 to {len(documents)}), is it relevant to the question? Return a list of {len(documents)} scores in order."""}
        ]
        
        result = client.invoke_structured(messages, GradeDocumentsBatch)
        
        # Map scores back to documents and filter
        relevant_docs = []
        scores = result.scores
        
        # Handle case where LLM returns wrong number of scores
        if len(scores) != len(documents):
            # Fallback: if we got fewer scores, pad with "no"; if more, truncate
            if len(scores) < len(documents):
                scores = scores + ["no"] * (len(documents) - len(scores))
            else:
                scores = scores[:len(documents)]
        
        for doc, score in zip(documents, scores):
            if score == "yes":
                relevant_docs.append(doc)
        
        # If no relevant docs, flag for web search
        needs_web_search = len(relevant_docs) == 0
        
        return relevant_docs, needs_web_search


class GenerationGrader:
    """
    Combined grader that performs hallucination and answer quality checks in a single LLM call.
    
    Uses mistral-small for good balance of speed and accuracy in grading.
    
    Returns:
        - "useful": Generation is grounded AND answers the question
        - "not supported": Generation contains hallucinations
        - "not useful": Generation is grounded but doesn't answer the question
    """
    
    SYSTEM_PROMPT = """You are a grader assessing the quality of an LLM-generated answer.

You must evaluate TWO criteria:

1. **Grounding (is_grounded)**: Is the answer grounded in / supported by the provided documents?
   - Check if the key claims in the answer are supported by the documents
   - Minor elaborations are OK, but core facts must be grounded
   - 'yes' if grounded, 'no' if it contains hallucinations or unsupported claims

2. **Usefulness (answers_question)**: Does the answer address the user's question?
   - Check if the answer directly addresses what was asked
   - The answer doesn't need to be perfect, just relevant to the question
   - 'yes' if useful, 'no' if it misses the point

Be strict on grounding (hallucinations are bad), but reasonable on usefulness (partial answers count)."""

    def __init__(self, model: str = GEN_GRADER_MODEL):
        self.model = model
        self._client = None
        
    def _get_client(self) -> MistralStructuredClient:
        if self._client is None:
            self._client = MistralStructuredClient(self.model)
        return self._client
    
    def grade(
        self, 
        documents: str, 
        generation: str, 
        question: str
    ) -> Literal["useful", "not supported", "not useful"]:
        """
        Perform full quality check on generation in a single LLM call.
        
        Args:
            documents: Source documents (concatenated)
            generation: The LLM's answer
            question: The user's question
            
        Returns:
            "useful", "not supported", or "not useful"
        """
        client = self._get_client()
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"""Source Documents:
{documents}

Question: {question}

Answer to evaluate:
{generation}

Evaluate the answer for grounding and usefulness."""}
        ]
        
        result = client.invoke_structured(messages, GradeGeneration)
        
        # Map result to return value
        if result.is_grounded == "no":
            return "not supported"
        
        if result.answers_question == "no":
            return "not useful"
        
        return "useful"


class QueryRewriter:
    """
    Rewrites user queries into standalone search queries using conversation history.
    
    Uses mistral-small for good balance of speed and quality.
    
    Transforms contextual queries like "What about Claude's score?" into
    standalone queries like "What is Claude 2's MMLU benchmark score?"
    """
    
    SYSTEM_PROMPT = """You are an expert at reformulating user questions into standalone search queries.

Context: The search targets NeurIPS 2025 academic papers on machine learning and AI.

Your task:
1. Given a conversation history and the user's latest question, rewrite it as a standalone query
2. The rewritten query should be self-contained - it should make sense without the conversation history
3. Preserve the user's original intent and any specific details mentioned
4. Expand abbreviations when helpful for search (e.g., "RL" → "reinforcement learning", "LLM" → "large language model")
5. If the question is already standalone, return it with minimal changes
6. Keep the query concise and search-friendly — remove filler words

Examples:
- History: "Tell me about GPT-4" / User: "What about its MMLU score?" → "GPT-4 MMLU benchmark performance"
- History: "Papers on RLHF" / User: "Any from DeepMind?" → "DeepMind reinforcement learning from human feedback RLHF"
- User: "What is RAG?" → "retrieval augmented generation RAG"
- User: "How do LLMs handle long context?" → "large language models long context handling techniques" """

    def __init__(self, model: str = REWRITER_MODEL):
        self.model = model
        self._client = None
        
    def _get_client(self) -> MistralStructuredClient:
        if self._client is None:
            self._client = MistralStructuredClient(self.model)
        return self._client
    
    def rewrite(
        self, 
        question: str, 
        history: Optional[List[dict]] = None
    ) -> RewrittenQuery:
        """
        Rewrite a question into a standalone search query.
        
        Args:
            question: The user's current question
            history: List of previous messages [{"role": "user"|"assistant", "content": "..."}]
            
        Returns:
            RewrittenQuery with the standalone query and reasoning
        """
        client = self._get_client()
        
        # Format history if provided
        history_text = ""
        if history:
            history_parts = []
            for msg in history[-6:]:  # Last 3 turns max (6 messages)
                role = msg.get("role", "user").upper()
                content = msg.get("content", "")[:MAX_HISTORY_LENGTH]  # Truncate long messages
                history_parts.append(f"{role}: {content}")
            history_text = "\n".join(history_parts)
        
        user_content = f"""Conversation History:
{history_text if history_text else "(No previous conversation)"}

Current User Question: {question}

Rewrite the question as a standalone search query."""

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        
        return client.invoke_structured(messages, RewrittenQuery)


class DocumentReranker:
    """
    Reranks retrieved documents by relevance to the query using Cohere Rerank API.
    
    Usage:
        reranker = DocumentReranker()
        reranked_docs = reranker.rerank(docs, query, top_k=10)
    """
    
    def __init__(
        self,
        cohere_model: str = "rerank-v3.5",
    ):
        """
        Initialize the reranker.
        
        Args:
            cohere_model: Cohere rerank model name
        """
        self.cohere_model = cohere_model
        self._cohere_client = None
    
    def _get_cohere_client(self):
        """Lazy init Cohere client."""
        if self._cohere_client is None:
            import cohere
            api_key = os.getenv("COHERE_API_KEY")
            if not api_key:
                raise ValueError("COHERE_API_KEY not found in environment")
            self._cohere_client = cohere.ClientV2(api_key=api_key)
        return self._cohere_client
    
    def rerank(
        self,
        documents: List[Any],
        query: str,
        top_k: int = 10,
    ) -> List[Any]:
        """
        Rerank documents by relevance to query using Cohere Rerank.
        
        Args:
            documents: List of Document objects (must have .content attribute)
            query: The search query
            top_k: Number of top documents to return
            
        Returns:
            List of top_k documents reordered by relevance (highest first)
        """
        if not documents:
            return []
        
        if len(documents) <= top_k:
            top_k = len(documents)
        
        client = self._get_cohere_client()
        
        # Extract text content from documents
        doc_texts = []
        for doc in documents:
            content = doc.content if hasattr(doc, 'content') else str(doc)
            doc_texts.append(content[:MAX_RERANK_LENGTH])  # Cohere has token limits
        
        response = client.rerank(
            model=self.cohere_model,
            query=query,
            documents=doc_texts,
            top_n=top_k,
        )
        
        # Reorder documents based on rerank results
        reranked = []
        for result in response.results:
            doc = documents[result.index]
            # Update score with rerank relevance score
            if hasattr(doc, 'score'):
                doc.score = result.relevance_score
            reranked.append(doc)
        
        return reranked
