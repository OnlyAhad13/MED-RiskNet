"""
Training utilities and trainer for MED-RiskNET.

Implements:
- Combined loss (classification + ranking)
- Training loop
- Checkpointing
- Seed setting
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
from pathlib import Path
from typing import Optional, Dict
import json


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CombinedLoss(nn.Module):
    """
    Combined loss for classification and ranking.
    
    Loss = BCE(classification) + ranking_weight * HingeLoss(ranking)
    
    Args:
        ranking_weight: Weight for ranking loss component
        margin: Margin for hinge loss
    """
    
    def __init__(self, ranking_weight: float = 0.5, margin: float = 1.0):
        super(CombinedLoss, self).__init__()
        self.ranking_weight = ranking_weight
        self.margin = margin
        self.bce = nn.BCELoss()
    
    def forward(
        self,
        class_pred: torch.Tensor,
        rank_scores: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            class_pred: Classification predictions [batch_size, 1]
            rank_scores: Ranking scores [batch_size, 1]
            labels: Ground truth labels [batch_size]
        
        Returns:
            Dictionary with total_loss, class_loss, rank_loss
        """
        # Classification loss (BCE)
        class_loss = self.bce(class_pred.squeeze(), labels)
        
        # Ranking loss (pairwise hinge)
        rank_loss = self._pairwise_hinge_loss(rank_scores.squeeze(), labels)
        
        # Combined loss
        total_loss = class_loss + self.ranking_weight * rank_loss
        
        return {
            'total': total_loss,
            'classification': class_loss,
            'ranking': rank_loss
        }
    
    def _pairwise_hinge_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Pairwise hinge loss for ranking.
        
        For pairs (i, j) where label[i] > label[j],
        enforce score[i] > score[j] + margin
        """
        # Find positive and negative pairs
        labels = labels.unsqueeze(1)
        label_diff = labels - labels.t()
        
        # Get score differences
        score_diff = scores.unsqueeze(1) - scores.unsqueeze(0)
        
        # Hinge loss: max(0, margin - score_diff) for positive pairs
        # Positive pair: label[i] > label[j], so we want score[i] > score[j]
        hinge = torch.clamp(self.margin - score_diff, min=0)
        
        # Only apply to pairs where labels differ
        mask = (label_diff > 0).float()
        
        # Average over valid pairs
        num_pairs = mask.sum() + 1e-8
        rank_loss = (hinge * mask).sum() / num_pairs
        
        return rank_loss


class MedRiskNetTrainer:
    """
    Trainer for MED-RiskNET model.
    
    Args:
        model: MedRiskNet model
        device: Device to train on
        lr: Learning rate
        weight_decay: Weight decay for AdamW
        ranking_loss_weight: Weight for ranking loss
        checkpoint_dir: Directory to save checkpoints
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        ranking_loss_weight: float = 0.5,
        checkpoint_dir: str = 'checkpoints'
    ):
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # Combined loss
        self.criterion = CombinedLoss(ranking_weight=ranking_loss_weight)
        
        # AdamW optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler (cosine annealing)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        # Training stats
        self.epoch = 0
        self.best_loss = float('inf')
        self.history = {
            'train_loss': [],
            'train_class_loss': [],
            'train_rank_loss': []
        }
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            verbose: Whether to print progress
        
        Returns:
            Dictionary with average losses
        """
        self.model.train()
        
        total_loss = 0
        class_loss_sum = 0
        rank_loss_sum = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Get data
            tabular = batch['tabular'].to(self.device)
            labels = batch['label'].to(self.device).float()
            
            # Optional modalities
            image = batch.get('image')
            if image is not None:
                image = image.to(self.device)
            
            text = batch.get('text')  # List of strings
            
            # Build graph if model uses GNN
            graph_x = None
            edge_index = None
            if self.model.use_graph:
                from data.graph_builder import PatientGraphBuilder
                graph_builder = PatientGraphBuilder(k_neighbors=3)
                edge_index = graph_builder.build_graph_from_features(tabular)
                graph_x = tabular  # Use patient features as node features
                edge_index = edge_index.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(
                tabular=tabular,
                image=image,
                text=text,
                graph_x=graph_x,
                edge_index=edge_index
            )
            
            # Compute loss
            losses = self.criterion(
                class_pred=outputs['classification'],
                rank_scores=outputs['ranking'],
                labels=labels
            )
            
            # Backward pass
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += losses['total'].item()
            class_loss_sum += losses['classification'].item()
            rank_loss_sum += losses['ranking'].item()
            num_batches += 1
            
            if verbose and (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}: "
                      f"Loss = {losses['total'].item():.4f}")
        
        # Average losses
        avg_losses = {
            'total': total_loss / num_batches,
            'classification': class_loss_sum / num_batches,
            'ranking': rank_loss_sum / num_batches
        }
        
        # Update history
        self.history['train_loss'].append(avg_losses['total'])
        self.history['train_class_loss'].append(avg_losses['classification'])
        self.history['train_rank_loss'].append(avg_losses['ranking'])
        
        return avg_losses
    
    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        save_every: int = 10,
        verbose: bool = True
    ):
        """
        Train for multiple epochs.
        
        Args:
            train_loader: Training data loader
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            verbose: Whether to print progress
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Checkpoint dir: {self.checkpoint_dir}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch + 1
            
            if verbose:
                print(f"\nEpoch {self.epoch}/{num_epochs}")
                print("-" * 50)
            
            # Train
            avg_losses = self.train_epoch(train_loader, verbose=verbose)
            
            # Step scheduler
            self.scheduler.step()
            
            if verbose:
                print(f"  Avg Total Loss: {avg_losses['total']:.4f}")
                print(f"  Avg Class Loss: {avg_losses['classification']:.4f}")
                print(f"  Avg Rank Loss: {avg_losses['ranking']:.4f}")
                print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            if self.epoch % save_every == 0 or self.epoch == num_epochs:
                self.save_checkpoint(f'epoch_{self.epoch}.pt')
            
            # Save best model
            if avg_losses['total'] < self.best_loss:
                self.best_loss = avg_losses['total']
                self.save_checkpoint('best_model.pt')
                if verbose:
                    print(f"  ✓ New best model saved!")
        
        print(f"\nTraining complete!")
        print(f"Best loss: {self.best_loss:.4f}")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'history': self.history
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.history = checkpoint['history']
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
