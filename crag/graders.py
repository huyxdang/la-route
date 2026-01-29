"""
LLM Graders with Structured Output for CRAG workflow.

Uses Pydantic models with Mistral Large for structured, reliable decisions:
1. Query Router - routes to vectorstore or web search
2. Document Grader - grades document relevance
3. Hallucination Grader - checks if generation is grounded
4. Answer Grader - checks if answer addresses the question
"""

import os
import json
from typing import Literal, List
from pydantic import BaseModel, Field

# Default model for all graders
DEFAULT_MODEL = "mistral-large-latest"


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


class GradeDocuments(BaseModel):
    """Binary relevance score for a document."""
    binary_score: Literal["yes", "no"] = Field(
        description="Document is relevant to the question: 'yes' or 'no'"
    )


class GradeHallucination(BaseModel):
    """Hallucination check result."""
    binary_score: Literal["yes", "no"] = Field(
        description="Answer is grounded in the documents: 'yes' if grounded, 'no' if hallucinated"
    )


class GradeAnswer(BaseModel):
    """Answer quality check result."""
    binary_score: Literal["yes", "no"] = Field(
        description="Answer addresses the question: 'yes' or 'no'"
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
        
        # Add JSON schema instruction to system message
        schema = response_model.model_json_schema()
        schema_str = json.dumps(schema, indent=2)
        
        # Prepend schema instruction
        enhanced_messages = messages.copy()
        if enhanced_messages and enhanced_messages[0]["role"] == "system":
            enhanced_messages[0]["content"] += f"""

You MUST respond with a valid JSON object matching this schema:
{schema_str}

Respond ONLY with the JSON object, no other text."""
        
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
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return response_model.model_validate(data)
            raise ValueError(f"Failed to parse structured response: {e}\nContent: {content}")


# ============== Grader Classes ==============

class QueryRouter:
    """
    Routes queries to appropriate datasource.
    
    - Technical papers, indexed content → vectorstore
    - General knowledge, current events → web_search
    """
    
    SYSTEM_PROMPT = """You are an expert at routing user questions to the appropriate data source.

You have access to two data sources:
1. **vectorstore**: Contains indexed technical documents, research papers, and domain-specific content. 
   Use this for questions about specific papers, technical concepts, indexed materials.
   
2. **web_search**: For general knowledge, current events, recent news, or topics not likely in the vectorstore.
   Use this for questions about recent developments, general facts, or when the query seems outside the indexed domain.

Based on the question, decide which source is most appropriate."""

    def __init__(self, model: str = DEFAULT_MODEL):
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
    Grades document relevance to a question.
    
    Checks if a document contains keywords or semantic meaning relevant to the question.
    """
    
    SYSTEM_PROMPT = """You are a grader assessing the relevance of a retrieved document to a user question.

Your task:
1. Check if the document contains keywords or semantic meaning relevant to the question
2. The document does NOT need to fully answer the question - just be relevant/useful
3. Give a binary 'yes' or 'no' score

Be lenient - if there's any reasonable connection, mark it as relevant."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self._client = None
        
    def _get_client(self) -> MistralStructuredClient:
        if self._client is None:
            self._client = MistralStructuredClient(self.model)
        return self._client
    
    def grade(self, document: str, question: str) -> GradeDocuments:
        """
        Grade a single document's relevance.
        
        Args:
            document: The document content
            question: The user's question
            
        Returns:
            GradeDocuments with binary_score
        """
        client = self._get_client()
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"""Document:
{document}

Question: {question}

Is this document relevant to the question?"""}
        ]
        
        return client.invoke_structured(messages, GradeDocuments)
    
    def grade_documents(
        self, 
        documents: List, 
        question: str
    ) -> tuple[List, bool]:
        """
        Grade multiple documents and filter irrelevant ones.
        
        Args:
            documents: List of Document objects
            question: The user's question
            
        Returns:
            Tuple of (filtered_documents, needs_web_search)
        """
        relevant_docs = []
        
        for doc in documents:
            content = doc.content if hasattr(doc, 'content') else str(doc)
            result = self.grade(content, question)
            
            if result.binary_score == "yes":
                relevant_docs.append(doc)
        
        # If no relevant docs (or too few), flag for web search
        needs_web_search = len(relevant_docs) == 0
        
        return relevant_docs, needs_web_search


class HallucinationGrader:
    """
    Checks if a generation is grounded in the provided documents.
    
    Identifies hallucinations - statements not supported by the source material.
    """
    
    SYSTEM_PROMPT = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.

Your task:
1. Check if the key claims in the generation are supported by the documents
2. Minor elaborations are OK, but core facts must be grounded
3. Give a binary 'yes' (grounded) or 'no' (hallucinated) score

Be strict - if there are unsupported factual claims, mark as 'no'."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self._client = None
        
    def _get_client(self) -> MistralStructuredClient:
        if self._client is None:
            self._client = MistralStructuredClient(self.model)
        return self._client
    
    def grade(self, documents: str, generation: str) -> GradeHallucination:
        """
        Check if generation is grounded in documents.
        
        Args:
            documents: The source documents (concatenated text)
            generation: The LLM's generation
            
        Returns:
            GradeHallucination with binary_score
        """
        client = self._get_client()
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"""Source Documents:
{documents}

Generation:
{generation}

Is this generation grounded in the documents?"""}
        ]
        
        return client.invoke_structured(messages, GradeHallucination)


class AnswerGrader:
    """
    Checks if an answer actually addresses the question.
    
    A generation might be factually correct but miss the point of the question.
    """
    
    SYSTEM_PROMPT = """You are a grader assessing whether an answer addresses / resolves a question.

Your task:
1. Check if the answer directly addresses what was asked
2. The answer doesn't need to be perfect, just relevant to the question
3. Give a binary 'yes' (useful) or 'no' (not useful) score

Be reasonable - partial answers that move toward resolution count as 'yes'."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self._client = None
        
    def _get_client(self) -> MistralStructuredClient:
        if self._client is None:
            self._client = MistralStructuredClient(self.model)
        return self._client
    
    def grade(self, question: str, generation: str) -> GradeAnswer:
        """
        Check if generation answers the question.
        
        Args:
            question: The user's question
            generation: The LLM's answer
            
        Returns:
            GradeAnswer with binary_score
        """
        client = self._get_client()
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"""Question: {question}

Answer: {generation}

Does this answer address the question?"""}
        ]
        
        return client.invoke_structured(messages, GradeAnswer)


class GenerationGrader:
    """
    Combined grader that performs both hallucination and answer quality checks.
    
    Returns:
        - "useful": Generation is grounded AND answers the question
        - "not supported": Generation contains hallucinations
        - "not useful": Generation is grounded but doesn't answer the question
    """
    
    def __init__(self, model: str = DEFAULT_MODEL):
        self.hallucination_grader = HallucinationGrader(model)
        self.answer_grader = AnswerGrader(model)
    
    def grade(
        self, 
        documents: str, 
        generation: str, 
        question: str
    ) -> Literal["useful", "not supported", "not useful"]:
        """
        Perform full quality check on generation.
        
        Args:
            documents: Source documents (concatenated)
            generation: The LLM's answer
            question: The user's question
            
        Returns:
            "useful", "not supported", or "not useful"
        """
        # Step 1: Check for hallucinations
        hallucination_result = self.hallucination_grader.grade(documents, generation)
        
        if hallucination_result.binary_score == "no":
            return "not supported"
        
        # Step 2: Check if answer is useful
        answer_result = self.answer_grader.grade(question, generation)
        
        if answer_result.binary_score == "no":
            return "not useful"
        
        return "useful"
