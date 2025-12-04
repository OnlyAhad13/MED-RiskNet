"""
Uncertainty estimation for MED-RiskNET using MC-Dropout.

Enables uncertainty quantification through multiple stochastic forward passes.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict


class MCDropoutWrapper(nn.Module):
    """
    Monte Carlo Dropout wrapper for uncertainty estimation.
    
    Enables dropout during inference and performs multiple forward passes
    to estimate prediction uncertainty.
    
    Args:
        model: Base model to wrap
        num_samples: Number of MC samples for uncertainty estimation
    """
    
    def __init__(self, model: nn.Module, num_samples: int = 20):
        super(MCDropoutWrapper, self).__init__()
        self.model = model
        self.num_samples = num_samples
    
    def enable_dropout(self):
        """Enable dropout layers during inference."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def forward_with_uncertainty(
        self,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with uncertainty estimation.
        
        Performs multiple forward passes with dropout enabled
        to compute mean and variance.
        
        Args:
            **kwargs: Model inputs (tabular, image, text, etc.)
        
        Returns:
            Dictionary with:
                - mean_classification: Mean predicted probability
                - std_classification: Standard deviation
                - mean_ranking: Mean ranking score
                - std_ranking: Standard deviation
                - all_predictions: All MC samples
        """
        # Enable dropout
        self.enable_dropout()
        
        # Collect predictions from multiple passes
        all_class_preds = []
        all_rank_preds = []
        
        for _ in range(self.num_samples):
            with torch.no_grad():
                outputs = self.model(**kwargs)
                all_class_preds.append(outputs['classification'])
                all_rank_preds.append(outputs['ranking'])
        
        # Stack predictions
        all_class_preds = torch.stack(all_class_preds)  # [num_samples, batch, 1]
        all_rank_preds = torch.stack(all_rank_preds)    # [num_samples, batch, 1]
        
        # Compute mean and std
        mean_class = all_class_preds.mean(dim=0)
        std_class = all_class_preds.std(dim=0)
        
        mean_rank = all_rank_preds.mean(dim=0)
        std_rank = all_rank_preds.std(dim=0)
        
        return {
            'mean_classification': mean_class,
            'std_classification': std_class,
            'mean_ranking': mean_rank,
            'std_ranking': std_rank,
            'all_classification': all_class_preds,
            'all_ranking': all_rank_preds
        }
    
    def predict_with_confidence(
        self,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get prediction with confidence interval.
        
        Args:
            **kwargs: Model inputs
        
        Returns:
            mean_prediction, confidence_interval (95%)
        """
        outputs = self.forward_with_uncertainty(**kwargs)
        
        mean = outputs['mean_classification']
        std = outputs['std_classification']
        
        # 95% confidence interval (1.96 * std)
        ci = 1.96 * std
        
        return mean, ci


def compute_prediction_entropy(predictions: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of MC-Dropout predictions (uncertainty measure).
    
    Higher entropy = more uncertain
    
    Args:
        predictions: MC samples [num_samples, batch, 1]
    
    Returns:
        Entropy values [batch]
    """
    # For binary classification
    mean_pred = predictions.mean(dim=0).squeeze()  # [batch]
    
    # Binary entropy: -p*log(p) - (1-p)*log(1-p)
    p = mean_pred.clamp(min=1e-7, max=1-1e-7)
    entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
    
    return entropy


def identify_uncertain_predictions(
    predictions: torch.Tensor,
    threshold: float = 0.1
) -> torch.Tensor:
    """
    Identify predictions with high uncertainty.
    
    Args:
        predictions: MC samples [num_samples, batch, 1]
        threshold: Std threshold for uncertainty
    
    Returns:
        Boolean mask [batch] - True for uncertain predictions
    """
    std = predictions.std(dim=0).squeeze()
    return std > threshold


__all__ = [
    'MCDropoutWrapper',
    'compute_prediction_entropy',
    'identify_uncertain_predictions'
]
