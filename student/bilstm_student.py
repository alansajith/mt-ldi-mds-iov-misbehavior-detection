"""
MT-LDI-MDS: BiLSTM Student Model
=================================
This module implements the lightweight student model for misbehavior detection.
The architecture is based on the base paper's BiLSTM design with:
- Input embedding layer
- Two Conv1D layers for local feature extraction
- Three stacked BiLSTM layers for temporal modeling
- Multi-head self-attention for global context
- Residual connection + LayerNorm
- Global Average Pooling
- Fully connected classifier

The model outputs both classification logits and intermediate features
for knowledge distillation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PermuteLayer(nn.Module):
    """
    Utility layer to permute tensor dimensions.
    Used to rearrange for LSTM input: (batch, channels, seq) -> (batch, seq, channels)
    """
    def __init__(self, dims: Tuple[int, ...]):
        super().__init__()
        self.dims = dims
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims)


class BiLSTMStudent(nn.Module):
    """
    BiLSTM Student Model for IoV Misbehavior Detection.
    
    Architecture (from base paper):
    1. Input Embedding: input_dim -> 128
    2. Conv1D Block 1: 128 -> 64, kernel=3, padding=same, ReLU
    3. Conv1D Block 2: 64 -> 64, kernel=3, padding=same, ReLU
    4. Permute: (batch, 64, seq_len) -> (batch, seq_len, 64)
    5. BiLSTM Layer 1: 64 -> 128 (bidirectional), dropout=0.3
    6. BiLSTM Layer 2: 256 -> 128 (bidirectional), dropout=0.3
    7. BiLSTM Layer 3: 256 -> 128 (bidirectional), dropout=0.3
    8. Multi-head Self-Attention: 256 dim, 4 heads
    9. Residual + LayerNorm
    10. Global Average Pooling: (batch, seq_len, 256) -> (batch, 256)
    11. FC Layer: 256 -> 128, BatchNorm, Dropout(0.3)
    12. Output Layer: 128 -> num_classes (9)
    
    The intermediate features (after FC BatchNorm, before output) are 128-dim
    and used for knowledge distillation.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 9,
        embed_dim: int = 128,
        conv_filters: int = 64,
        conv_kernel: int = 3,
        lstm_hidden: int = 128,
        lstm_layers: int = 3,
        lstm_dropout: float = 0.3,
        attention_heads: int = 4,
        attention_dim: int = 256,
        fc_hidden: int = 128,
        fc_dropout: float = 0.3,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the BiLSTM Student model.
        
        Args:
            input_dim: Input feature dimension (number of BSM features)
            num_classes: Number of attack classes (default 9)
            embed_dim: Embedding dimension (default 128)
            conv_filters: Number of Conv1D filters (default 64)
            conv_kernel: Conv1D kernel size (default 3)
            lstm_hidden: LSTM hidden size per direction (default 128)
            lstm_layers: Number of stacked BiLSTM layers (default 3)
            lstm_dropout: Dropout between LSTM layers (default 0.3)
            attention_heads: Number of attention heads (default 4)
            attention_dim: Attention embedding dimension (default 256)
            fc_hidden: FC layer hidden dimension (default 128)
            fc_dropout: FC layer dropout (default 0.3)
            device: Torch device (cuda/cpu)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.lstm_hidden = lstm_hidden
        self.attention_dim = attention_dim
        self.fc_hidden = fc_hidden
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"Initializing BiLSTMStudent on {self.device}")
        print(f"  Input dim: {input_dim}")
        print(f"  Num classes: {num_classes}")
        print(f"  Embed dim: {embed_dim}")
        print(f"  LSTM hidden: {lstm_hidden} (bidirectional -> {lstm_hidden*2})")
        print(f"  Attention dim: {attention_dim}, heads: {attention_heads}")
        print(f"  FC hidden: {fc_hidden}")
        
        # ====================================================================
        # 1. Input Embedding Layer: input_dim -> embed_dim (128)
        # ====================================================================
        self.input_embedding = nn.Linear(input_dim, embed_dim)
        
        # ====================================================================
        # 2-3. Two Conv1D Layers with ReLU
        # Input: (batch, embed_dim, seq_len) after permute
        # Conv1D expects (batch, channels, seq_len)
        # ====================================================================
        self.conv1 = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=conv_filters,
            kernel_size=conv_kernel,
            padding='same',
            bias=True
        )
        self.conv1_bn = nn.BatchNorm1d(conv_filters)
        self.conv1_relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(
            in_channels=conv_filters,
            out_channels=conv_filters,
            kernel_size=conv_kernel,
            padding='same',
            bias=True
        )
        self.conv2_bn = nn.BatchNorm1d(conv_filters)
        self.conv2_relu = nn.ReLU(inplace=True)
        
        # ====================================================================
        # 4. Permute for LSTM: (batch, channels, seq) -> (batch, seq, channels)
        # ====================================================================
        self.permute = PermuteLayer((0, 2, 1))  # (batch, seq_len, conv_filters)
        
        # ====================================================================
        # 5-7. Three Stacked BiLSTM Layers
        # Each BiLSTM: input -> hidden*2 (bidirectional)
        # Layer 1: conv_filters(64) -> lstm_hidden*2(256)
        # Layer 2: 256 -> 256
        # Layer 3: 256 -> 256
        # ====================================================================
        self.bilstm1 = nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0  # Dropout handled between layers
        )
        
        self.bilstm2 = nn.LSTM(
            input_size=lstm_hidden * 2,  # 256 from bidirectional
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )
        
        self.bilstm3 = nn.LSTM(
            input_size=lstm_hidden * 2,  # 256
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )
        
        # Dropout between LSTM layers
        self.lstm_dropout = nn.Dropout(lstm_dropout)
        
        # ====================================================================
        # 8. Multi-head Self-Attention
        # Input: (batch, seq_len, 256) -> Output: (batch, seq_len, 256)
        # ====================================================================
        self.self_attention = nn.MultiheadAttention(
            embed_dim=attention_dim,  # 256
            num_heads=attention_heads,  # 4
            batch_first=True,
            dropout=0.1
        )
        
        # ====================================================================
        # 9. Residual Connection + LayerNorm
        # ====================================================================
        self.attn_layernorm = nn.LayerNorm(attention_dim)
        
        # ====================================================================
        # 10. Global Average Pooling
        # (batch, seq_len, 256) -> (batch, 256)
        # ====================================================================
        # Implemented in forward with mean(dim=1)
        
        # ====================================================================
        # 11. FC Layer: 256 -> 128 with BatchNorm and Dropout
        # This output (128-dim) is used for knowledge distillation
        # ====================================================================
        self.fc = nn.Linear(attention_dim, fc_hidden)
        self.fc_bn = nn.BatchNorm1d(fc_hidden)
        self.fc_relu = nn.ReLU(inplace=True)
        self.fc_dropout = nn.Dropout(fc_dropout)
        
        # ====================================================================
        # 12. Output Layer: 128 -> num_classes
        # ====================================================================
        self.output_layer = nn.Linear(fc_hidden, num_classes)
        
        # Initialize weights
        self._init_weights()
        
        # Move to device
        self.to(self.device)
        
        # Print parameter count
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the BiLSTM student.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim) or (batch, input_dim)
               If 2D, treats as single timestep and adds sequence dimension.
               
        Returns:
            Tuple of (logits, intermediate_features)
            - logits: (batch, num_classes) classification output
            - intermediate_features: (batch, fc_hidden=128) features before output layer
        """
        # Handle 2D input (batch, input_dim) -> add sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
        
        batch_size, seq_len, _ = x.shape
        
        # Move to device
        x = x.to(self.device)
        
        # ====================================================================
        # 1. Input Embedding: (batch, seq_len, input_dim) -> (batch, seq_len, embed_dim)
        # ====================================================================
        x = self.input_embedding(x)  # (batch, seq_len, 128)
        
        # ====================================================================
        # 2-3. Conv1D Layers (need channels-first: batch, channels, seq)
        # ====================================================================
        x = x.permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.conv1_relu(x)
        
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = self.conv2_relu(x)
        # x shape: (batch, conv_filters=64, seq_len)
        
        # ====================================================================
        # 4. Permute for LSTM: (batch, channels, seq) -> (batch, seq, channels)
        # ====================================================================
        x = self.permute(x)  # (batch, seq_len, 64)
        
        # ====================================================================
        # 5-7. Three Stacked BiLSTM Layers
        # ====================================================================
        # BiLSTM 1: (batch, seq_len, 64) -> (batch, seq_len, 256)
        x, _ = self.bilstm1(x)
        x = self.lstm_dropout(x)
        
        # BiLSTM 2: (batch, seq_len, 256) -> (batch, seq_len, 256)
        x, _ = self.bilstm2(x)
        x = self.lstm_dropout(x)
        
        # BiLSTM 3: (batch, seq_len, 256) -> (batch, seq_len, 256)
        x, _ = self.bilstm3(x)
        # x shape: (batch, seq_len, 256)
        
        # ====================================================================
        # 8. Multi-head Self-Attention
        # ====================================================================
        attn_out, _ = self.self_attention(x, x, x)  # (batch, seq_len, 256)
        
        # ====================================================================
        # 9. Residual + LayerNorm
        # ====================================================================
        x = self.attn_layernorm(attn_out + x)  # Residual connection
        
        # ====================================================================
        # 10. Global Average Pooling: (batch, seq_len, 256) -> (batch, 256)
        # ====================================================================
        x = x.mean(dim=1)  # (batch, 256)
        
        # ====================================================================
        # 11. FC Layer: 256 -> 128 with BatchNorm, ReLU, Dropout
        # ====================================================================
        intermediate = self.fc(x)  # (batch, 128)
        intermediate = self.fc_bn(intermediate)
        intermediate = self.fc_relu(intermediate)
        intermediate = self.fc_dropout(intermediate)
        # intermediate shape: (batch, 128) - THIS IS USED FOR KD
        
        # ====================================================================
        # 12. Output Layer: 128 -> num_classes
        # ====================================================================
        logits = self.output_layer(intermediate)  # (batch, num_classes)
        
        return logits, intermediate
    
    def get_intermediate_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get intermediate features (128-dim) for knowledge distillation.
        This is the output of FC layer after BatchNorm, before the final classifier.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim) or (batch, input_dim)
            
        Returns:
            Intermediate features (batch, 128)
        """
        _, intermediate = self.forward(x)
        return intermediate
    
    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get only classification logits.
        
        Args:
            x: Input tensor
            
        Returns:
            Logits (batch, num_classes)
        """
        logits, _ = self.forward(x)
        return logits


def create_student(
    input_dim: int,
    num_classes: int = 9,
    device: Optional[torch.device] = None,
    **kwargs
) -> BiLSTMStudent:
    """
    Factory function to create a BiLSTMStudent with default paper parameters.
    
    Args:
        input_dim: Input feature dimension
        num_classes: Number of classes (default 9)
        device: Torch device
        **kwargs: Additional arguments to override defaults
        
    Returns:
        BiLSTMStudent instance
    """
    defaults = {
        'embed_dim': 128,
        'conv_filters': 64,
        'conv_kernel': 3,
        'lstm_hidden': 128,
        'lstm_layers': 3,
        'lstm_dropout': 0.3,
        'attention_heads': 4,
        'attention_dim': 256,
        'fc_hidden': 128,
        'fc_dropout': 0.3,
    }
    defaults.update(kwargs)
    
    return BiLSTMStudent(
        input_dim=input_dim,
        num_classes=num_classes,
        device=device,
        **defaults
    )


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing BiLSTMStudent...")
    
    # Test parameters
    batch_size = 8
    seq_len = 10  # Number of BSM messages in a sequence
    input_dim = 20  # Number of features per BSM
    num_classes = 9
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_student(input_dim=input_dim, num_classes=num_classes, device=device)
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, input_dim).to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits, intermediate = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Intermediate features shape: {intermediate.shape}")
    
    # Test get_intermediate_features
    features = model.get_intermediate_features(x)
    print(f"get_intermediate_features shape: {features.shape}")
    assert torch.allclose(intermediate, features), "Features should match!"
    
    # Test classify
    logits_only = model.classify(x)
    print(f"classify shape: {logits_only.shape}")
    assert torch.allclose(logits, logits_only), "Logits should match!"
    
    # Test gradient flow
    print("\n--- Gradient Test ---")
    model.train()
    logits, intermediate = model(x)
    loss = logits.sum() + intermediate.sum()
    loss.backward()
    print(f"Gradients computed successfully")
    print(f"Input embedding grad norm: {model.input_embedding.weight.grad.norm().item():.4f}")
    print(f"Output layer grad norm: {model.output_layer.weight.grad.norm().item():.4f}")
    
    print("\nAll tests passed!")