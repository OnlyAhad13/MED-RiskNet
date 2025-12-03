"""
Multimodal fusion models for MED-RiskNET.

Combines embeddings from multiple modalities (tabular, image, text, graph)
using transformer-based fusion mechanisms.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
import math


class FusionTransformer(nn.Module):
    """
    Multimodal fusion using transformer architecture.
    
    Accepts embeddings from different modalities, projects them to a common
    dimension, adds modality-specific positional embeddings, and fuses them
    using self-attention.
    
    Architecture:
        1. Project each modality to common dimension
        2. Add learned modality-type embeddings
        3. Stack into sequence: [tabular, image, text, graph]
        4. Apply transformer encoder
        5. Aggregate (mean/max pooling or CLS token)
    
    Args:
        modality_dims: Dictionary mapping modality name -> input dimension
                      e.g., {'tabular': 32, 'image': 512, 'text': 768, 'graph': 16}
        hidden_dim: Common dimension for fusion
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout probability
        fusion_method: How to aggregate ('mean', 'max', 'cls')
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        fusion_method: str = 'mean'
    ):
        super(FusionTransformer, self).__init__()
        
        self.modality_dims = modality_dims
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.fusion_method = fusion_method
        
        # Modality projection layers
        self.projections = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for modality, dim in modality_dims.items()
        })
        
        # Modality-type embeddings (learned positional encodings)
        self.modality_embeddings = nn.ParameterDict({
            modality: nn.Parameter(torch.randn(1, hidden_dim))
            for modality in modality_dims.keys()
        })
        
        # Optional CLS token for aggregation
        if fusion_method == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        tabular: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        graph: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through fusion transformer.
        
        Args:
            tabular: Tabular embeddings [batch_size, tabular_dim]
            image: Image embeddings [batch_size, image_dim]
            text: Text embeddings [batch_size, text_dim]
            graph: Graph embeddings [batch_size, graph_dim]
        
        Returns:
            Fused embeddings [batch_size, hidden_dim]
        """
        batch_size = None
        tokens = []
        modality_names = []
        
        # Project each modality and add modality embeddings
        for modality_name, embedding in [
            ('tabular', tabular),
            ('image', image),
            ('text', text),
            ('graph', graph)
        ]:
            if embedding is not None:
                if batch_size is None:
                    batch_size = embedding.shape[0]
                
                # Project to common dimension
                projected = self.projections[modality_name](embedding)  # [batch, hidden_dim]
                
                # Add modality-type embedding
                modality_emb = self.modality_embeddings[modality_name]  # [1, hidden_dim]
                projected = projected + modality_emb  # Broadcasting
                
                # Add to sequence
                tokens.append(projected.unsqueeze(1))  # [batch, 1, hidden_dim]
                modality_names.append(modality_name)
        
        if len(tokens) == 0:
            raise ValueError("At least one modality must be provided")
        
        # Stack tokens into sequence
        token_sequence = torch.cat(tokens, dim=1)  # [batch, num_modalities, hidden_dim]
        
        # Add CLS token if using CLS aggregation
        if self.fusion_method == 'cls':
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, hidden_dim]
            token_sequence = torch.cat([cls_tokens, token_sequence], dim=1)
        
        # Apply transformer
        fused = self.transformer(token_sequence)  # [batch, seq_len, hidden_dim]
        
        # Aggregate
        if self.fusion_method == 'mean':
            output = fused.mean(dim=1)  # [batch, hidden_dim]
        elif self.fusion_method == 'max':
            output = fused.max(dim=1)[0]  # [batch, hidden_dim]
        elif self.fusion_method == 'cls':
            output = fused[:, 0, :]  # [batch, hidden_dim] - CLS token
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        # Final projection
        output = self.output_proj(output)
        
        return output


class CrossModalAttentionFusion(nn.Module):
    """
    Cross-modal attention fusion.
    
    Uses one modality as query and others as key/value for cross-attention.
    Can be configured to use any modality as primary.
    
    Args:
        modality_dims: Dictionary of modality dimensions
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        primary_modality: Which modality to use as query ('tabular', 'image', 'text', 'graph')
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_heads: int = 8,
        primary_modality: str = 'tabular',
        dropout: float = 0.1
    ):
        super(CrossModalAttentionFusion, self).__init__()
        
        self.modality_dims = modality_dims
        self.hidden_dim = hidden_dim
        self.primary_modality = primary_modality
        
        # Project all modalities
        self.projections = nn.ModuleDict({
            modality: nn.Linear(dim, hidden_dim)
            for modality, dim in modality_dims.items()
        })
        
        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(
        self,
        tabular: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        graph: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with cross-modal attention.
        
        Args:
            tabular, image, text, graph: Modality embeddings
        
        Returns:
            Fused embedding [batch_size, hidden_dim]
        """
        # Collect all modality embeddings
        modalities = {
            'tabular': tabular,
            'image': image,
            'text': text,
            'graph': graph
        }
        
        # Project all available modalities
        projected = {}
        for name, emb in modalities.items():
            if emb is not None:
                projected[name] = self.projections[name](emb).unsqueeze(1)  # [batch, 1, hidden_dim]
        
        if len(projected) == 0:
            raise ValueError("At least one modality must be provided")
        
        # Primary modality is query
        if self.primary_modality not in projected:
            raise ValueError(f"Primary modality '{self.primary_modality}' not provided")
        
        query = projected[self.primary_modality]  # [batch, 1, hidden_dim]
        
        # Other modalities are key/value
        kv_list = [emb for name, emb in projected.items() if name != self.primary_modality]
        if len(kv_list) > 0:
            key_value = torch.cat(kv_list, dim=1)  # [batch, num_other_modalities, hidden_dim]
        else:
            # Only primary modality available
            key_value = query
        
        # Cross-attention
        attended, _ = self.cross_attention(
            query=query,
            key=key_value,
            value=key_value
        )  # [batch, 1, hidden_dim]
        
        # Squeeze and process
        output = attended.squeeze(1)  # [batch, hidden_dim]
        output = self.output_mlp(output)
        
        return output


class MultimodalClassifier(nn.Module):
    """
    Complete multimodal classifier with fusion and classification head.
    
    Args:
        modality_dims: Dictionary of modality dimensions
        hidden_dim: Fusion hidden dimension
        num_classes: Number of output classes
        fusion_type: 'transformer' or 'cross_attention'
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_classes: int = 1,
        fusion_type: str = 'transformer',
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super(MultimodalClassifier, self).__init__()
        
        self.num_classes = num_classes
        
        # Fusion module
        if fusion_type == 'transformer':
            self.fusion = FusionTransformer(
                modality_dims=modality_dims,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                fusion_method='mean'
            )
        elif fusion_type == 'cross_attention':
            self.fusion = CrossModalAttentionFusion(
                modality_dims=modality_dims,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Output activation
        if num_classes == 1:
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.Softmax(dim=1)
    
    def forward(
        self,
        tabular: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        graph: Optional[torch.Tensor] = None,
        return_embedding: bool = False
    ):
        """
        Forward pass through multimodal classifier.
        
        Args:
            tabular, image, text, graph: Modality embeddings
            return_embedding: If True, return both predictions and fused embedding
        
        Returns:
            Predictions (and optionally fused embedding)
        """
        # Fuse modalities
        fused = self.fusion(
            tabular=tabular,
            image=image,
            text=text,
            graph=graph
        )
        
        # Classify
        logits = self.classifier(fused)
        predictions = self.output_activation(logits)
        
        if return_embedding:
            return predictions, fused
        return predictions
