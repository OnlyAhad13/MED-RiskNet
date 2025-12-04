"""
Model calibration utilities for MED-RiskNET.

Implements temperature scaling to improve probability calibration.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Optional


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for probability calibration.
    
    Applies a single scalar temperature parameter to logits:
        calibrated_prob = softmax(logits / temperature)
    
    For binary classification:
        calibrated_prob = sigmoid(logits / temperature)
    
    Args:
        initial_temperature: Initial temperature value
    """
    
    def __init__(self, initial_temperature: float = 1.5):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * initial_temperature)
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Raw model outputs [batch_size, ...]
        
        Returns:
            Scaled logits [batch_size, ...]
        """
        return logits / self.temperature
    
    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50
    ):
        """
        Fit temperature on validation set.
        
        Args:
            logits: Validation logits [n_samples, ...]
            labels: True labels [n_samples]
            lr: Learning rate for optimization
            max_iter: Maximum optimization iterations
        """
        # Use NLL loss for calibration
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = criterion(scaled_logits, labels)
            loss.backward()
            return loss
        
        # Optimize temperature
        optimizer.step(eval_loss)
        
        # Ensure temperature is positive
        with torch.no_grad():
            self.temperature.clamp_(min=0.01)
        
        return self.temperature.item()
    
    def calibrate_probabilities(
        self,
        logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Get calibrated probabilities from logits.
        
        Args:
            logits: Raw logits [batch_size, ...]
        
        Returns:
            Calibrated probabilities [batch_size, ...]
        """
        scaled_logits = self.forward(logits)
        return torch.sigmoid(scaled_logits)


def calibrate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: str = 'cpu'
) -> TemperatureScaling:
    """
    Fit temperature scaling on validation set.
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        device: Device to run on
    
    Returns:
        Fitted TemperatureScaling module
    """
    model.eval()
    
    # Collect validation logits and labels
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            tabular = batch['tabular'].to(device)
            labels = batch['label'].to(device)
            
            # Optional modalities
            image = batch.get('image')
            if image is not None:
                image = image.to(device)
            text = batch.get('text')
            
            # Forward pass
            outputs = model(tabular=tabular, image=image, text=text)
            
            # Get logits (before sigmoid)
            # For binary: logits = inverse_sigmoid(probs)
            probs = outputs['classification']
            logits = torch.logit(probs.clamp(min=1e-7, max=1-1e-7))
            
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    
    # Concatenate all batches
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0).unsqueeze(1)
    
    # Fit temperature
    temp_scaling = TemperatureScaling()
    optimal_temp = temp_scaling.fit(all_logits, all_labels)
    
    print(f"Optimal temperature: {optimal_temp:.4f}")
    
    return temp_scaling


__all__ = ['TemperatureScaling', 'calibrate_model']
