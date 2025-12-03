"""
Text encoder models for MED-RiskNET using pretrained transformers.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List

class TextEncoder(nn.Module):
    """
    Text encoder using pretrained clinical BERT models.
    
    Supports:
    - BioClinicalBERT (emilyalsentzer/Bio_ClinicalBERT) - Primary choice
    - BioBERT - Alternative
    - BERT-base - Fallback
    
    Args:
        model_name: Model identifier from HuggingFace
        feature_dim: Output feature dimension (768 default for BERT)
        max_length: Maximum sequence length for tokenization
        freeze_bert: Whether to freeze BERT weights
    """
    
    def __init__(
        self,
        model_name: str = 'emilyalsentzer/Bio_ClinicalBERT',
        feature_dim: int = 768,
        max_length: int = 512,
        freeze_bert: bool = False
    ):
        super(TextEncoder, self).__init__()
        
        self.model_name = model_name
        self.feature_dim = feature_dim
        self.max_length = max_length
        
        # Try to load transformers
        try:
            from transformers import AutoModel, AutoTokenizer
            
            # Load tokenizer
            print(f"Loading tokenizer: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load BERT model
            print(f"Loading model: {model_name}")
            self.bert = AutoModel.from_pretrained(model_name)
            
            # Get actual BERT output dimension
            bert_dim = self.bert.config.hidden_size
            print(f"Using {model_name} (BERT dim: {bert_dim})")
            
        except Exception as e:
            print(f"Warning: Could not load {model_name}: {e}")
            print("Creating placeholder encoder (for testing without transformers)")
            
            # Placeholder for testing without transformers
            self.tokenizer = None
            self.bert = None
            bert_dim = 768  # Standard BERT dimension
        
        # Freeze BERT if requested
        if freeze_bert and self.bert is not None:
            for param in self.bert.parameters():
                param.requires_grad = False
            print(f"BERT backbone frozen (parameters not trainable)")
        
        # Projection head (if feature_dim != bert_dim)
        if feature_dim != bert_dim:
            self.projection = nn.Sequential(
                nn.Linear(bert_dim, feature_dim),
                nn.Tanh(),
                nn.Dropout(0.1)
            )
        else:
            self.projection = nn.Identity()
    
    def tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize text inputs.
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary with input_ids, attention_mask, etc.
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Install transformers: pip install transformers")
        
        # Tokenize
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return encoding
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Forward pass through the text encoder.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            texts: Raw text strings (will be tokenized if provided)
            
        Returns:
            Feature vectors of shape (batch_size, feature_dim)
        """
        # If texts provided, tokenize them first
        if texts is not None:
            encoding = self.tokenize(texts)
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
        
        if self.bert is None:
            raise RuntimeError("BERT model not loaded. Install transformers: pip install transformers")
        
        # Move to same device as model
        device = next(self.bert.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Extract CLS token embedding (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch_size, bert_dim]
        
        # Project to desired dimension
        features = self.projection(cls_embedding)  # [batch_size, feature_dim]
        
        return features
    
    def get_embedding_dim(self) -> int:
        """Return the output embedding dimension."""
        return self.feature_dim


class TextClassifier(nn.Module):
    """
    Complete text classifier with encoder and classification head.
    
    Args:
        model_name: BERT model identifier
        feature_dim: Intermediate feature dimension
        num_classes: Number of output classes (1 for binary)
        max_length: Maximum sequence length
        freeze_bert: Whether to freeze BERT backbone
    """
    
    def __init__(
        self,
        model_name: str = 'emilyalsentzer/Bio_ClinicalBERT',
        feature_dim: int = 768,
        num_classes: int = 1,
        max_length: int = 512,
        freeze_bert: bool = False
    ):
        super(TextClassifier, self).__init__()
        
        self.num_classes = num_classes
        
        # Text encoder
        self.encoder = TextEncoder(
            model_name=model_name,
            feature_dim=feature_dim,
            max_length=max_length,
            freeze_bert=freeze_bert
        )
        
        # Classification head
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Output activation
        if num_classes == 1:
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.Softmax(dim=1)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
        return_embedding: bool = False
    ):
        """
        Forward pass through the text classifier.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            texts: Raw text strings
            return_embedding: If True, return both predictions and embeddings
            
        Returns:
            Predictions (and optionally embeddings)
        """
        # Get embeddings
        embeddings = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            texts=texts
        )
        
        # Get predictions
        logits = self.classifier(embeddings)
        predictions = self.output_activation(logits)
        
        if return_embedding:
            return predictions, embeddings
        return predictions
