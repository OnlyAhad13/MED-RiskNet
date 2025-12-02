"""
Tabular data models for MED-RiskNet
"""

import torch 
import torch.nn as nn 
from typing import List, Optional

class TabularEncoder(nn.Module):
    """
    Multi-layer perceptron encoder for tabular clinical data.
    
    Architecture:
        Input -> [Linear -> BatchNorm -> Activation -> Dropout] x N -> Output
    
    Args:
        input_dim: Dimensionality of input features
        hidden_dims: List of hidden layer dimensions
        output_dim: Dimensionality of output embedding
        dropout: Dropout probability
        activation: Activation function ('relu', 'gelu', 'elu')
        use_batch_norm: Whether to use batch normalization
    """

    def __init__(
        self, 
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        output_dim: int = 32,
        dropout: float = 0.3,
        activation: str = 'relu',
        use_batch_norm: bool = True
    ):
        super(TabularEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dims[i+1]))
            
            if i < len(dims)-2:
                layers.append(self.activation)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)

    def get_embedding_dim(self) -> int:
        """Return the output embedding dimension."""
        return self.output_dim

class TabularClassifier(nn.Module):
    """
    Complete tabular classifier with encoder and classification head.
    
    Args:
        input_dim: Dimensionality of input features
        hidden_dims: List of hidden layer dimensions for encoder
        num_classes: Number of output classes (1 for binary)
        dropout: Dropout probability
        activation: Activation function
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        num_classes: int = 1,
        dropout: float = 0.3,
        activation: str = 'relu'
    ):
        super(TabularClassifier, self).__init__()

        self.num_classes = num_classes

        # Encoder
        self.encoder = TabularEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=hidden_dims[-1] if hidden_dims else 32,
            dropout=dropout,
            activation=activation
        )

        #Classification Head
        self.classifier = nn.Linear(self.encoder.output_dim, num_classes)
        # For binary classification
        if num_classes == 1:
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.Softmax(dim=1)
        
    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        """
        Forward pass through the classifier.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            return_embedding: If True, return both predictions and embeddings
            
        Returns:
            Predictions (and optionally embeddings)
        """

        #Get embeddings
        embeddings = self.encoder(x)

        #Get predictions
        logits = self.classifier(embeddings)
        predictions = self.output_activation(logits)

        if return_embedding:
            return predictions, embeddings
        return predictions
        
class ResidualBlock(nn.Module):
    """
    Residual block for deeper tabular networks.
    
    Args:
        dim: Feature dimension
        dropout: Dropout probability
    """

    def __init__(self, dim: int, dropout: float = 0.3):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        return self.activation(x + self.block(x))

class ResidualTabularEncoder(nn.Module):
    """
    Tabular encoder with residual connections for deeper networks.
    
    Args:
        input_dim: Dimensionality of input features
        hidden_dim: Hidden dimension (kept constant)
        num_blocks: Number of residual blocks
        output_dim: Dimensionality of output embedding
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        output_dim: int = 64,
        dropout: float = 0.3
    ): 
        super(ResidualTabularEncoder, self).__init__()

        self.output_dim = output_dim
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout)
            for _ in range(num_blocks)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual encoder."""
        x = self.input_proj(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.output_proj(x)
        return x
    
    def get_embedding_dim(self) -> int:
        """Return the output embedding dimension."""
        return self.output_dim
