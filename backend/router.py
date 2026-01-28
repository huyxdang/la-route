"""
Model routing module for Le-Route.
Supports rule-based fallback and trained MLP router.
"""

import os
import pickle
from typing import Optional
from dataclasses import dataclass
import numpy as np

from .prompts import detect_risk_level, is_complex_query
from .models import RiskLevel


# Model identifiers
MODEL_SMALL = "ministral-8b-latest"
MODEL_LARGE = "mistral-large-latest"

# Cost per million tokens (USD)
COST_PER_MILLION = {
    MODEL_SMALL: {"input": 0.10, "output": 0.10},
    MODEL_LARGE: {"input": 2.00, "output": 6.00},
}


@dataclass
class RoutingDecision:
    """Result of a routing decision."""
    model: str
    confidence: float
    risk_level: RiskLevel
    reason: str
    features: Optional[dict] = None


class RuleBasedRouter:
    """
    Simple rule-based router for fallback.
    Uses keyword detection and query complexity.
    """
    
    def __init__(self):
        self.name = "rule-based"
    
    def route(
        self,
        query: str,
        top_similarity: float,
        num_relevant_chunks: int,
        doc_type: str
    ) -> RoutingDecision:
        """
        Make routing decision based on rules.
        
        Args:
            query: User's question
            top_similarity: Highest chunk similarity score
            num_relevant_chunks: Number of chunks with similarity > 0.5
            doc_type: Document type (policy, contract, legal, technical, general)
        
        Returns:
            RoutingDecision with model choice and reasoning
        """
        risk = detect_risk_level(query)
        is_complex = is_complex_query(query)
        
        # Convert string risk to enum
        risk_enum = RiskLevel(risk)
        
        # Decision logic
        reasons = []
        use_large = False
        confidence = 0.8  # Base confidence
        
        # High risk queries always go to large model
        if risk == "high":
            use_large = True
            reasons.append("high-risk query detected")
            confidence = 0.95
        
        # Complex queries go to large model
        elif is_complex:
            use_large = True
            reasons.append("complex query structure")
            confidence = 0.85
        
        # Low similarity means harder retrieval - use large model
        elif top_similarity < 0.5:
            use_large = True
            reasons.append(f"low retrieval confidence ({top_similarity:.2f})")
            confidence = 0.75
        
        # Few relevant chunks might mean complex answer needed
        elif num_relevant_chunks < 2 and doc_type in ["legal", "contract"]:
            use_large = True
            reasons.append("limited context in legal/contract doc")
            confidence = 0.70
        
        # Medium risk with technical docs
        elif risk == "medium" and doc_type == "technical":
            use_large = True
            reasons.append("medium risk in technical document")
            confidence = 0.75
        
        # Default to small model
        else:
            reasons.append("standard query")
            confidence = 0.85
        
        model = MODEL_LARGE if use_large else MODEL_SMALL
        reason = "; ".join(reasons)
        
        return RoutingDecision(
            model=model,
            confidence=confidence,
            risk_level=risk_enum,
            reason=reason
        )


class MLPRouter:
    """
    Trained MLP router for optimal routing decisions.
    Uses features extracted from query and retrieval context.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        pca_path: Optional[str] = None
    ):
        self.name = "mlp"
        self.model = None
        self.pca = None
        self.loaded = False
        
        # Try to load model and PCA
        model_path = model_path or os.getenv("MLP_MODEL_PATH", "./mlp_router.pt")
        pca_path = pca_path or os.getenv("PCA_PATH", "./pca.pkl")
        
        self._load_model(model_path, pca_path)
    
    def _load_model(self, model_path: str, pca_path: str):
        """Load the trained MLP model and PCA transformer."""
        try:
            import torch
            from mlp.model import RoutingMLP
            
            if os.path.exists(model_path) and os.path.exists(pca_path):
                # Load model
                self.model = RoutingMLP()
                self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
                self.model.eval()
                
                # Load PCA
                with open(pca_path, "rb") as f:
                    self.pca = pickle.load(f)
                
                self.loaded = True
                print(f"✓ MLP router loaded from {model_path}")
            else:
                print(f"⚠ MLP model not found at {model_path}, using rule-based fallback")
        except Exception as e:
            print(f"⚠ Failed to load MLP router: {e}, using rule-based fallback")
    
    def extract_features(
        self,
        query_embedding: list[float],
        top_similarities: list[float],
        query_token_count: int,
        doc_token_count: int,
        doc_type: str
    ) -> np.ndarray:
        """
        Extract features for MLP input.
        
        Returns:
            numpy array of 25 features
        """
        features = []
        
        # 1. Query embedding compressed via PCA (16 dims)
        query_vec = np.array(query_embedding).reshape(1, -1)
        if self.pca is not None:
            query_pca = self.pca.transform(query_vec).flatten()
        else:
            # Fallback: take first 16 dims
            query_pca = query_vec.flatten()[:16]
        features.extend(query_pca.tolist())
        
        # 2. Top chunk similarity (1 dim)
        top_sim = max(top_similarities) if top_similarities else 0.0
        features.append(top_sim)
        
        # 3. Number of relevant chunks (similarity > 0.5) (1 dim)
        num_relevant = sum(1 for s in top_similarities if s > 0.5)
        features.append(float(num_relevant))
        
        # 4. Query token count normalized (1 dim)
        features.append(query_token_count / 100.0)
        
        # 5. Doc token count normalized (1 dim)
        features.append(doc_token_count / 10000.0)
        
        # 6. Doc type one-hot (5 dims)
        doc_types = ["policy", "contract", "legal", "technical", "general"]
        onehot = [1.0 if doc_type == dt else 0.0 for dt in doc_types]
        features.extend(onehot)
        
        return np.array(features, dtype=np.float32)
    
    def route(
        self,
        query: str,
        query_embedding: list[float],
        top_similarities: list[float],
        query_token_count: int,
        doc_token_count: int,
        doc_type: str
    ) -> RoutingDecision:
        """
        Make routing decision using trained MLP.
        Falls back to rule-based if MLP not loaded.
        """
        import torch
        
        # Get risk level for the response
        risk = detect_risk_level(query)
        risk_enum = RiskLevel(risk)
        
        if not self.loaded:
            # Fallback to rule-based
            fallback = RuleBasedRouter()
            decision = fallback.route(
                query=query,
                top_similarity=max(top_similarities) if top_similarities else 0.0,
                num_relevant_chunks=sum(1 for s in top_similarities if s > 0.5),
                doc_type=doc_type
            )
            decision.reason = f"[fallback] {decision.reason}"
            return decision
        
        # Extract features
        features = self.extract_features(
            query_embedding=query_embedding,
            top_similarities=top_similarities,
            query_token_count=query_token_count,
            doc_token_count=doc_token_count,
            doc_type=doc_type
        )
        
        # Run through MLP
        with torch.no_grad():
            x = torch.tensor(features).unsqueeze(0)
            probs = self.model(x).squeeze().numpy()
        
        p_small, p_large = probs[0], probs[1]
        
        # Conservative threshold: use Large if >30% chance it's needed
        if p_large > 0.3:
            model = MODEL_LARGE
            confidence = float(p_large)
            reason = f"MLP: P(large)={p_large:.2f} > 0.3 threshold"
        else:
            model = MODEL_SMALL
            confidence = float(p_small)
            reason = f"MLP: P(small)={p_small:.2f}, P(large)={p_large:.2f}"
        
        return RoutingDecision(
            model=model,
            confidence=confidence,
            risk_level=risk_enum,
            reason=reason,
            features={
                "p_small": float(p_small),
                "p_large": float(p_large),
                "top_similarity": float(max(top_similarities) if top_similarities else 0),
                "num_relevant": sum(1 for s in top_similarities if s > 0.5)
            }
        )


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int
) -> float:
    """
    Estimate cost in USD for a query.
    
    Args:
        model: Model identifier
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
    
    Returns:
        Estimated cost in USD
    """
    if model not in COST_PER_MILLION:
        return 0.0
    
    costs = COST_PER_MILLION[model]
    input_cost = (input_tokens / 1_000_000) * costs["input"]
    output_cost = (output_tokens / 1_000_000) * costs["output"]
    
    return input_cost + output_cost


def get_router(router_type: Optional[str] = None):
    """
    Get the appropriate router based on configuration.
    
    Args:
        router_type: "mlp" or "rule". Defaults to env ROUTER_TYPE or "rule".
    
    Returns:
        Router instance (MLPRouter or RuleBasedRouter)
    """
    router_type = router_type or os.getenv("ROUTER_TYPE", "rule")
    
    if router_type == "mlp":
        router = MLPRouter()
        if router.loaded:
            return router
        # Fall back to rule-based if MLP failed to load
        print("⚠ MLP router not available, using rule-based")
    
    return RuleBasedRouter()
