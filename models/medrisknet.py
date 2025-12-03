"""
Complete MED-RiskNET model integrating all modality encoders.

This module wires together:
- TabularEncoder
- CNNEncoder  
- TextEncoder
- GNNEncoder
- FusionTransformer
- Classification and Ranking Heads
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

from models.tabular import TabularEncoder
from models.image import CNNEncoder
from models.text import TextEncoder
from models.graph import GNNEncoder
from models.fusion import FusionTransformer


class MedRiskNet(nn.Module):
    """
    Complete multimodal health risk prediction model.
    
    Integrates:
    - Tabular clinical data
    - Medical images (X-rays)
    - Clinical text reports
    - Patient similarity graphs
    
    Outputs:
    - Classification: Risk probability [0, 1]
    - Ranking: Pairwise comparison scores
    
    Args:
        tabular_input_dim: Dimension of tabular features
        num_classes: Number of output classes (1 for binary)
        fusion_hidden_dim: Hidden dimension for fusion
        use_graph: Whether to use graph encoder
        freeze_encoders: Whether to freeze pretrained encoders
    """
    
    def __init__(
        self,
        tabular_input_dim: int,
        num_classes: int = 1,
        fusion_hidden_dim: int = 256,
        use_graph: bool = False,
        freeze_encoders: bool = False
    ):
        super(MedRiskNet, self).__init__()
        
        self.num_classes = num_classes
        self.use_graph = use_graph
        
        # Tabular Encoder
        self.tabular_encoder = TabularEncoder(
            input_dim=tabular_input_dim,
            hidden_dims=[128, 64],
            output_dim=32,
            dropout=0.3
        )
        
        # Image Encoder
        self.image_encoder = CNNEncoder(
            model_name='resnet50',
            pretrained=True,
            feature_dim=512,
            freeze_backbone=freeze_encoders
        )
        
        # Text Encoder
        try:
            self.text_encoder = TextEncoder(
                model_name='emilyalsentzer/Bio_ClinicalBERT',
                feature_dim=768,
                max_length=512,
                freeze_bert=freeze_encoders
            )
            self.has_text_encoder = True
        except:
            print("Warning: Text encoder not available (transformers not installed)")
            self.text_encoder = None
            self.has_text_encoder = False
        
        # Graph Encoder (optional)
        if use_graph:
            self.graph_encoder = GNNEncoder(
                input_dim=tabular_input_dim,
                hidden_dims=[64, 32],
                output_dim=16,
                gnn_type='sage'
            )
        else:
            self.graph_encoder = None
        
        # Fusion Transformer
        modality_dims = {
            'tabular': 32,
            'image': 512
        }
        if self.has_text_encoder:
            modality_dims['text'] = 768
        if use_graph:
            modality_dims['graph'] = 16
        
        self.fusion = FusionTransformer(
            modality_dims=modality_dims,
            hidden_dim=fusion_hidden_dim,
            num_heads=8,
            num_layers=2,
            dropout=0.1,
            fusion_method='mean'
        )
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # Ranking Head
        self.ranker = nn.Sequential(
            nn.Linear(fusion_hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # Ranking score
        )
        
        # Output activations
        if num_classes == 1:
            self.classification_activation = nn.Sigmoid()
        else:
            self.classification_activation = nn.Softmax(dim=1)
    
    def forward(
        self,
        tabular: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        text: Optional[list] = None,
        graph_x: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through MedRiskNet.
        
        Args:
            tabular: Tabular features [batch_size, input_dim]
            image: Image tensors [batch_size, 3, 224, 224]
            text: List of text strings
            graph_x: Graph node features
            edge_index: Graph edge indices
            return_embeddings: Whether to return intermediate embeddings
        
        Returns:
            Dictionary with:
                - 'classification': Class probabilities [batch_size, num_classes]
                - 'ranking': Ranking scores [batch_size, 1]
                - 'embeddings': Fused embeddings (if return_embeddings=True)
        """
        # Encode each modality
        tabular_emb = self.tabular_encoder(tabular)
        
        image_emb = None
        if image is not None:
            image_emb = self.image_encoder(image)
        
        text_emb = None
        if text is not None and self.has_text_encoder:
            text_emb = self.text_encoder(texts=text)
        
        graph_emb = None
        if self.use_graph and graph_x is not None and edge_index is not None:
            # Get per-node embeddings
            node_embs = self.graph_encoder(graph_x, edge_index)
            # Aggregate to batch-level (mean pooling)
            graph_emb = node_embs.mean(dim=0, keepdim=True)
            if tabular.shape[0] > 1:
                graph_emb = graph_emb.expand(tabular.shape[0], -1)
        
        # Fuse modalities
        fused = self.fusion(
            tabular=tabular_emb,
            image=image_emb,
            text=text_emb,
            graph=graph_emb
        )
        
        # Classification head
        class_logits = self.classifier(fused)
        class_probs = self.classification_activation(class_logits)
        
        # Ranking head
        rank_scores = self.ranker(fused)
        
        # Prepare output
        output = {
            'classification': class_probs,
            'ranking': rank_scores
        }
        
        if return_embeddings:
            output['embeddings'] = fused
            output['modality_embeddings'] = {
                'tabular': tabular_emb,
                'image': image_emb,
                'text': text_emb,
                'graph': graph_emb
            }
        
        return output
    
    def predict(
        self,
        tabular: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        text: Optional[list] = None,
        graph_x: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict risk scores (convenience method).
        
        Returns:
            Risk probabilities [batch_size, num_classes]
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(
                tabular=tabular,
                image=image,
                text=text,
                graph_x=graph_x,
                edge_index=edge_index
            )
        return output['classification']
