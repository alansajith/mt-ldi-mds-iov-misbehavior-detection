"""
MT-LDI-MDS: Weighted Aggregator Module
=======================================
This module implements the WeightedAggregator class that combines embeddings from
three specialized teacher models (Teacher A, B, C) into a unified representation
for knowledge distillation to the student model.

Two aggregation modes:
1. Fixed: Simple average ê = (eA + eB + eC) / 3
2. Learnable: ê = softmax([λ1, λ2, λ3]) · [eA, eB, eC] where λ are learnable parameters

After weighted sum, passes through a LinearAdapter (3*hidden_dim → student_hidden_dim).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List


class WeightedAggregator(nn.Module):
    """
    Weighted Aggregator for Multi-Teacher Knowledge Distillation.
    
    Combines embeddings from three specialized teachers:
    - Teacher A: DoS and Sybil attacks
    - Teacher B: Position spoofing attacks
    - Teacher C: Speed and replay attacks
    
    Two modes:
    - fixed: Equal weighting (1/3 each)
    - learnable: Softmax-weighted combination with learnable λ parameters
    
    The aggregated embedding is then projected to student's hidden dimension
    via a LinearAdapter.
    """
    
    def __init__(
        self,
        teacher_hidden_dim: int,
        student_hidden_dim: int = 256,
        mode: str = "learnable",
        init_weights: Optional[List[float]] = None
    ):
        """
        Initialize the WeightedAggregator.
        
        Args:
            teacher_hidden_dim: Hidden dimension of each teacher's embeddings
            student_hidden_dim: Target hidden dimension for student (default 256)
            mode: Aggregation mode - "fixed" or "learnable" (default "learnable")
            init_weights: Initial weights for learnable mode [λ1, λ2, λ3]
                         If None, initializes to [1/3, 1/3, 1/3]
        """
        super().__init__()
        
        self.teacher_hidden_dim = teacher_hidden_dim
        self.student_hidden_dim = student_hidden_dim
        self.mode = mode
        
        if mode not in ["fixed", "learnable"]:
            raise ValueError(f"Mode must be 'fixed' or 'learnable', got '{mode}'")
        
        # Learnable weight parameters (λ1, λ2, λ3)
        if mode == "learnable":
            if init_weights is None:
                init_weights = [1.0/3.0, 1.0/3.0, 1.0/3.0]
            # Initialize as logits (before softmax)
            init_logits = torch.log(torch.tensor(init_weights, dtype=torch.float32) + 1e-8)
            self.lambda_params = nn.Parameter(init_logits)
        else:
            # Fixed mode - no learnable parameters
            self.register_buffer("fixed_weights", torch.tensor([1.0/3.0, 1.0/3.0, 1.0/3.0]))
        
        # Linear adapter: 3 * teacher_hidden_dim -> student_hidden_dim
        # Concatenates [eA, eB, eC] then projects
        self.adapter = nn.Linear(3 * teacher_hidden_dim, student_hidden_dim)
        
        # LayerNorm for stability
        self.layer_norm = nn.LayerNorm(student_hidden_dim)
        
        # Initialize adapter weights
        nn.init.xavier_uniform_(self.adapter.weight)
        nn.init.zeros_(self.adapter.bias)
        
        print(f"WeightedAggregator initialized:")
        print(f"  Mode: {mode}")
        print(f"  Teacher hidden dim: {teacher_hidden_dim}")
        print(f"  Student hidden dim: {student_hidden_dim}")
        if mode == "learnable":
            print(f"  Initial λ weights: {self.get_weights()}")
    
    def get_weights(self) -> Dict[str, float]:
        """
        Get current aggregation weights.
        
        Returns:
            Dictionary with teacher names and their weights
        """
        if self.mode == "learnable":
            # Apply softmax to get normalized weights
            weights = F.softmax(self.lambda_params, dim=0).detach().cpu().numpy()
        else:
            weights = self.fixed_weights.detach().cpu().numpy()
        
        return {
            "teacher_A": float(weights[0]),
            "teacher_B": float(weights[1]),
            "teacher_C": float(weights[2])
        }
    
    def get_lambda_logits(self) -> torch.Tensor:
        """Get raw lambda logits (for learnable mode)."""
        if self.mode == "learnable":
            return self.lambda_params
        else:
            return torch.log(self.fixed_weights + 1e-8)
    
    def forward(
        self,
        eA: torch.Tensor,
        eB: torch.Tensor,
        eC: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: aggregate teacher embeddings and project to student dimension.
        
        Args:
            eA: Teacher A embeddings (batch, teacher_hidden_dim)
            eB: Teacher B embeddings (batch, teacher_hidden_dim)
            eC: Teacher C embeddings (batch, teacher_hidden_dim)
            
        Returns:
            Tuple of (aggregated_embedding, weights)
            - aggregated_embedding: (batch, student_hidden_dim)
            - weights: (3,) current aggregation weights
        """
        batch_size = eA.shape[0]
        device = eA.device
        
        # Get aggregation weights
        if self.mode == "learnable":
            # Softmax over lambda parameters
            weights = F.softmax(self.lambda_params, dim=0)  # (3,)
        else:
            weights = self.fixed_weights.to(device)  # (3,)
        
        # Weighted combination: ê = w1*eA + w2*eB + w3*eC
        # Expand weights for broadcasting: (3,) -> (1, 3, 1)
        w = weights.view(1, 3, 1)  # (1, 3, 1)
        
        # Stack embeddings: (batch, 3, teacher_hidden_dim)
        stacked = torch.stack([eA, eB, eC], dim=1)
        
        # Weighted sum: (batch, teacher_hidden_dim)
        weighted_sum = (stacked * w).sum(dim=1)
        
        # Also create concatenated version for adapter: (batch, 3*teacher_hidden_dim)
        concatenated = torch.cat([eA, eB, eC], dim=1)
        
        # Pass through linear adapter
        adapted = self.adapter(concatenated)  # (batch, student_hidden_dim)
        
        # LayerNorm for stability
        output = self.layer_norm(adapted)
        
        return output, weights
    
    def forward_concat_only(
        self,
        eA: torch.Tensor,
        eB: torch.Tensor,
        eC: torch.Tensor
    ) -> torch.Tensor:
        """
        Alternative forward: just concatenate and adapt (no weighted sum).
        Used when we want the adapter to learn the combination.
        """
        concatenated = torch.cat([eA, eB, eC], dim=1)
        adapted = self.adapter(concatenated)
        output = self.layer_norm(adapted)
        return output


class LinearAdapter(nn.Module):
    """
    Simple linear adapter module for dimension projection.
    Can be used standalone if needed.
    """
    
    def __init__(self, input_dim: int, output_dim: int, use_layernorm: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.layer_norm = nn.LayerNorm(output_dim)
        
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.use_layernorm:
            x = self.layer_norm(x)
        return x


def create_aggregator(
    teacher_hidden_dim: int,
    student_hidden_dim: int = 256,
    mode: str = "learnable"
) -> WeightedAggregator:
    """
    Factory function to create a WeightedAggregator.
    
    Args:
        teacher_hidden_dim: Hidden dimension of teacher embeddings
        student_hidden_dim: Target student hidden dimension
        mode: "fixed" or "learnable"
        
    Returns:
        WeightedAggregator instance
    """
    return WeightedAggregator(
        teacher_hidden_dim=teacher_hidden_dim,
        student_hidden_dim=student_hidden_dim,
        mode=mode
    )


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    # Quick test
    print("Testing WeightedAggregator...")
    
    # Test parameters
    batch_size = 4
    teacher_dim = 4096  # Qwen3-8B hidden dim
    student_dim = 256
    
    # Create dummy embeddings
    eA = torch.randn(batch_size, teacher_dim)
    eB = torch.randn(batch_size, teacher_dim)
    eC = torch.randn(batch_size, teacher_dim)
    
    # Test fixed mode
    print("\n--- Fixed Mode ---")
    agg_fixed = WeightedAggregator(teacher_dim, student_dim, mode="fixed")
    out_fixed, weights_fixed = agg_fixed(eA, eB, eC)
    print(f"Output shape: {out_fixed.shape}")
    print(f"Weights: {agg_fixed.get_weights()}")
    
    # Test learnable mode
    print("\n--- Learnable Mode ---")
    agg_learnable = WeightedAggregator(teacher_dim, student_dim, mode="learnable")
    out_learnable, weights_learnable = agg_learnable(eA, eB, eC)
    print(f"Output shape: {out_learnable.shape}")
    print(f"Initial weights: {agg_learnable.get_weights()}")
    
    # Test gradient flow
    print("\n--- Gradient Test ---")
    loss = out_learnable.sum()
    loss.backward()
    print(f"Lambda grad: {agg_learnable.lambda_params.grad}")
    print(f"Adapter weight grad norm: {agg_learnable.adapter.weight.grad.norm().item():.4f}")
    
    print("\nAll tests passed!")