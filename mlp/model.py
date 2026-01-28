"""
MLP Router model for Le-Route.
Predicts whether to use small (8B) or large (123B) model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class RoutingMLP(nn.Module):
    """
    Simple MLP for routing decisions.
    
    Input: 25 features
    - Query embedding (PCA compressed): 16 dims
    - Top chunk similarity: 1 dim
    - Num relevant chunks: 1 dim
    - Query token count (normalized): 1 dim
    - Doc token count (normalized): 1 dim
    - Doc type one-hot: 5 dims
    
    Output: 2 classes [P(small), P(large)]
    """
    
    def __init__(self, input_dim: int = 25, hidden_dims: list[int] = [64, 32], dropout: float = 0.1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 2))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Softmax probabilities of shape (batch_size, 2)
        """
        logits = self.net(x)
        return F.softmax(logits, dim=-1)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.3) -> torch.Tensor:
        """
        Make routing predictions.
        
        Args:
            x: Input tensor
            threshold: Use large model if P(large) > threshold
        
        Returns:
            Tensor of predictions (0=small, 1=large)
        """
        probs = self.forward(x)
        p_large = probs[:, 1]
        return (p_large > threshold).long()


def save_model(model: RoutingMLP, path: str):
    """Save model state dict."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"✓ Model saved to {path}")


def load_model(path: str, device: str = "cpu") -> RoutingMLP:
    """Load model from state dict."""
    model = RoutingMLP()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"✓ Model loaded from {path}")
    return model


if __name__ == "__main__":
    # Test model
    print("Testing RoutingMLP...")
    
    model = RoutingMLP()
    print(f"Model architecture:\n{model}")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params:,}")
    
    # Test forward pass
    batch = torch.randn(4, 25)  # 4 samples, 25 features
    output = model(batch)
    print(f"\nInput shape: {batch.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (probabilities):\n{output}")
    
    # Test prediction
    preds = model.predict(batch, threshold=0.3)
    print(f"\nPredictions (threshold=0.3): {preds.tolist()}")
    
    # Test save/load
    save_model(model, "test_model.pt")
    loaded = load_model("test_model.pt")
    
    # Verify same output
    with torch.no_grad():
        output2 = loaded(batch)
        assert torch.allclose(output, output2)
        print("✓ Save/load verification passed")
    
    # Cleanup
    Path("test_model.pt").unlink()
    print("\n✅ All tests passed!")
