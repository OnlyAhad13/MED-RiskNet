"""
Graph Neural Network encoders for MED-RiskNET.

Implements GNN models for learning from patient similarity graphs.
"""

import torch
import torch.nn as nn
from typing import Optional, List


class GNNEncoder(nn.Module):
    """
    Graph Neural Network encoder using PyTorch Geometric.
    
    Supports:
    - GraphSAGE (Hamilton et al., 2017) - Primary choice
    - GAT (Graph Attention Networks) - Alternative
    
    Args:
        input_dim: Input feature dimension
        hidden_dims: List of hidden dimensions
        output_dim: Output embedding dimension
        gnn_type: Type of GNN ('sage' or 'gat')
        dropout: Dropout probability
        num_heads: Number of attention heads (for GAT only)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        output_dim: int = 16,
        gnn_type: str = 'sage',
        dropout: float = 0.3,
        num_heads: int = 4
    ):
        super(GNNEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.gnn_type = gnn_type
        self.dropout = dropout
        
        # Try to import PyTorch Geometric
        try:
            if gnn_type == 'sage':
                from torch_geometric.nn import SAGEConv
                Conv = SAGEConv
                conv_kwargs = {}
            elif gnn_type == 'gat':
                from torch_geometric.nn import GATConv
                Conv = GATConv
                conv_kwargs = {'heads': num_heads, 'concat': False}
            else:
                raise ValueError(f"Unsupported GNN type: {gnn_type}")
            
            # Build GNN layers
            self.convs = nn.ModuleList()
            dims = [input_dim] + hidden_dims + [output_dim]
            
            for i in range(len(dims) - 1):
                self.convs.append(Conv(dims[i], dims[i+1], **conv_kwargs))
            
            self.activation = nn.ReLU()
            self.dropout_layer = nn.Dropout(dropout)
            
            print(f"Using {gnn_type.upper()} with {len(self.convs)} layers")
            
        except ImportError as e:
            print(f"Warning: PyTorch Geometric not available: {e}")
            print("Creating placeholder GNN (for testing without PyG)")
            print("Install with: pip install torch-geometric")
            
            # Placeholder MLP (simulates GNN without graph structure)
            self.convs = None
            layers = []
            dims = [input_dim] + hidden_dims + [output_dim]
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                if i < len(dims) - 2:
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout))
            self.mlp = nn.Sequential(*layers)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the GNN.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
        
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        if self.convs is None:
            # Placeholder mode (no PyG)
            return self.mlp(x)
        
        # GNN forward pass
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            # Apply activation and dropout (except last layer)
            if i < len(self.convs) - 1:
                x = self.activation(x)
                x = self.dropout_layer(x)
        
        return x
    
    def get_embedding_dim(self) -> int:
        """Return the output embedding dimension."""
        return self.output_dim


class GNNClassifier(nn.Module):
    """
    Complete GNN classifier with encoder and classification head.
    
    Args:
        input_dim: Input feature dimension
        hidden_dims: Hidden dimensions for GNN
        output_dim: GNN output dimension
        num_classes: Number of output classes (1 for binary)
        gnn_type: Type of GNN ('sage' or 'gat')
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        output_dim: int = 16,
        num_classes: int = 1,
        gnn_type: str = 'sage',
        dropout: float = 0.3
    ):
        super(GNNClassifier, self).__init__()
        
        self.num_classes = num_classes
        
        # GNN encoder
        self.encoder = GNNEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            gnn_type=gnn_type,
            dropout=dropout
        )
        
        # Classification head
        self.classifier = nn.Linear(output_dim, num_classes)
        
        # Output activation
        if num_classes == 1:
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.Softmax(dim=1)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_embedding: bool = False
    ):
        """
        Forward pass through the GNN classifier.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            return_embedding: If True, return both predictions and embeddings
        
        Returns:
            Predictions (and optionally embeddings)
        """
        # Get node embeddings
        embeddings = self.encoder(x, edge_index)
        
        # Get predictions
        logits = self.classifier(embeddings)
        predictions = self.output_activation(logits)
        
        if return_embedding:
            return predictions, embeddings
        return predictions


class HeterogeneousGNN(nn.Module):
    """
    Heterogeneous GNN for multi-modal patient graphs.
    
    Handles graphs with different node types:
    - Patient nodes
    - Feature nodes
    - Image nodes
    - Text nodes
    
    Uses message passing between different node types.
    
    Args:
        node_type_dims: Dictionary mapping node type -> feature dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        num_layers: Number of GNN layers
    """
    
    def __init__(
        self,
        node_type_dims: dict,
        hidden_dim: int = 64,
        output_dim: int = 32,
        num_layers: int = 2
    ):
        super(HeterogeneousGNN, self).__init__()
        
        self.node_type_dims = node_type_dims
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Project different node types to common dimension
        self.projections = nn.ModuleDict({
            node_type: nn.Linear(dim, hidden_dim)
            for node_type, dim in node_type_dims.items()
        })
        
        # GNN encoder on unified representation
        self.gnn = GNNEncoder(
            input_dim=hidden_dim,
            hidden_dims=[hidden_dim] * (num_layers - 1),
            output_dim=output_dim,
            gnn_type='sage'
        )
    
    def forward(
        self,
        x_dict: dict,
        edge_index: torch.Tensor,
        node_type_mapping: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for heterogeneous graph.
        
        Args:
            x_dict: Dictionary of features per node type
            edge_index: Edge indices [2, num_edges]
            node_type_mapping: Maps node index -> node type
        
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        # Project all nodes to common dimension
        num_nodes = sum(x.shape[0] for x in x_dict.values())
        x_unified = torch.zeros(num_nodes, self.hidden_dim)
        
        offset = 0
        for node_type, x in x_dict.items():
            n = x.shape[0]
            x_unified[offset:offset+n] = self.projections[node_type](x)
            offset += n
        
        # Apply GNN
        embeddings = self.gnn(x_unified, edge_index)
        
        return embeddings
