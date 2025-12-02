"""
Image encoder models for MED-RiskNET using pretrained CNNs.
"""

import torch
import torch.nn as nn
from typing import Optional

class CNNEncoder(nn.Module):
    """
    CNN-based image encoder using pretrained models.
    
    Supports:
    - EfficientNetV2 (via timm) - Primary choice
    - ResNet50 (via torchvision) - Fallback
    
    Args:
        model_name: Model architecture ('efficientnet_v2' or 'resnet50')
        pretrained: Whether to use ImageNet pretrained weights
        feature_dim: Output feature dimension
        freeze_backbone: Whether to freeze pretrained weights
    """
    
    def __init__(
        self,
        model_name: str = 'efficientnet_v2',
        pretrained: bool = True,
        feature_dim: int = 512,
        freeze_backbone: bool = False
    ):
        super(CNNEncoder, self).__init__()
        
        self.model_name = model_name
        self.feature_dim = feature_dim
        
        # Try to use timm for EfficientNetV2, fallback to torchvision ResNet
        if model_name == 'efficientnet_v2':
            try:
                import timm
                # EfficientNetV2-S (small variant)
                self.backbone = timm.create_model(
                    'tf_efficientnetv2_s',
                    pretrained=pretrained,
                    num_classes=0,  # Remove classification head
                    global_pool='avg'  # Global average pooling
                )
                backbone_out_dim = self.backbone.num_features
                print(f"✓ Using EfficientNetV2 from timm (output dim: {backbone_out_dim})")
            except ImportError:
                print("timm not available, falling back to ResNet50")
                model_name = 'resnet50'
        
        if model_name == 'resnet50':
            from torchvision import models
            if pretrained:
                backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            else:
                backbone = models.resnet50(weights=None)
            
            # Remove final FC layer
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            backbone_out_dim = 2048  # ResNet50 output dimension
            print(f"Using ResNet50 from torchvision (output dim: {backbone_out_dim})")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"Backbone frozen (parameters not trainable)")
        
        # Projection head to desired feature dimension
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_out_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self._init_projection()
    
    def _init_projection(self):
        """Initialize projection layer weights."""
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN encoder.
        
        Args:
            x: Input image tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Feature vector of shape (batch_size, feature_dim)
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Project to desired dimension
        embeddings = self.projection(features)
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Return the output embedding dimension."""
        return self.feature_dim


class ImageClassifier(nn.Module):
    """
    Complete image classifier with CNN encoder and classification head.
    
    Args:
        model_name: CNN architecture to use
        pretrained: Whether to use pretrained weights
        feature_dim: Intermediate feature dimension
        num_classes: Number of output classes (1 for binary)
        freeze_backbone: Whether to freeze CNN backbone
    """
    
    def __init__(
        self,
        model_name: str = 'efficientnet_v2',
        pretrained: bool = True,
        feature_dim: int = 512,
        num_classes: int = 1,
        freeze_backbone: bool = False
    ):
        super(ImageClassifier, self).__init__()
        
        self.num_classes = num_classes
        
        # CNN Encoder
        self.encoder = CNNEncoder(
            model_name=model_name,
            pretrained=pretrained,
            feature_dim=feature_dim,
            freeze_backbone=freeze_backbone
        )
        
        # Classification head
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Output activation
        if num_classes == 1:
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.Softmax(dim=1)
    
    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        """
        Forward pass through the image classifier.
        
        Args:
            x: Input image tensor of shape (batch_size, 3, 224, 224)
            return_embedding: If True, return both predictions and embeddings
            
        Returns:
            Predictions (and optionally embeddings)
        """
        # Get embeddings
        embeddings = self.encoder(x)
        
        # Get predictions
        logits = self.classifier(embeddings)
        predictions = self.output_activation(logits)
        
        if return_embedding:
            return predictions, embeddings
        return predictions
