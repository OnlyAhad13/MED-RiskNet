"""
Evaluation metrics for MED-RiskNET.

Implements:
- Classification metrics: AUROC, AUPRC, Precision/Recall/F1
- Ranking metrics: Recall@K, NDCG
- Calibration metrics: ECE (Expected Calibration Error)
"""

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    precision_recall_curve
)
from typing import Dict, Optional


def compute_auroc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Area Under ROC Curve.
    
    Args:
        y_true: True binary labels [n_samples]
        y_pred: Predicted probabilities [n_samples]
    
    Returns:
        AUROC score [0, 1]
    """
    return roc_auc_score(y_true, y_pred)


def compute_auprc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Area Under Precision-Recall Curve.
    
    Args:
        y_true: True binary labels [n_samples]
        y_pred: Predicted probabilities [n_samples]
    
    Returns:
        AUPRC score [0, 1]
    """
    return average_precision_score(y_true, y_pred)


def compute_precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute Precision, Recall, and F1 score.
    
    Args:
        y_true: True binary labels [n_samples]
        y_pred: Predicted probabilities [n_samples]
        threshold: Classification threshold
    
    Returns:
        Dictionary with precision, recall, f1
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred_binary,
        average='binary',
        zero_division=0
    )
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def compute_recall_at_k(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    k: int
) -> float:
    """
    Compute Recall@K (fraction of positives in top-K predictions).
    
    Args:
        y_true: True binary labels [n_samples]
        y_scores: Prediction scores [n_samples]
        k: Number of top predictions to consider
    
    Returns:
        Recall@K score [0, 1]
    """
    # Get indices of top-k predictions
    top_k_indices = np.argsort(y_scores)[-k:]
    
    # Count positives in top-k
    positives_in_top_k = y_true[top_k_indices].sum()
    
    # Total positives
    total_positives = y_true.sum()
    
    if total_positives == 0:
        return 0.0
    
    return positives_in_top_k / total_positives


def compute_ndcg(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    k: Optional[int] = None
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG@K).
    
    Measures ranking quality - higher scores for correctly ordered predictions.
    
    Args:
        y_true: True relevance scores [n_samples]
        y_scores: Predicted scores [n_samples]
        k: Number of top results to consider (None = all)
    
    Returns:
        NDCG score [0, 1]
    """
    # Sort by predicted scores
    order = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[order]
    
    if k is not None:
        y_true_sorted = y_true_sorted[:k]
    
    # Compute DCG (Discounted Cumulative Gain)
    gains = 2 ** y_true_sorted - 1
    discounts = np.log2(np.arange(len(y_true_sorted)) + 2)
    dcg = np.sum(gains / discounts)
    
    # Compute ideal DCG (if predictions were perfect)
    y_true_ideal = np.sort(y_true)[::-1]
    if k is not None:
        y_true_ideal = y_true_ideal[:k]
    
    ideal_gains = 2 ** y_true_ideal - 1
    ideal_discounts = np.log2(np.arange(len(y_true_ideal)) + 2)
    idcg = np.sum(ideal_gains / ideal_discounts)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def compute_ece(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Measures calibration - how well predicted probabilities match true frequencies.
    
    Args:
        y_true: True binary labels [n_samples]
        y_pred: Predicted probabilities [n_samples]
        n_bins: Number of bins for calibration
    
    Returns:
        ECE score (lower is better)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        # Find predictions in this bin
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)
        
        if not np.any(in_bin):
            continue
        
        # Compute accuracy and confidence in bin
        bin_accuracy = y_true[in_bin].mean()
        bin_confidence = y_pred[in_bin].mean()
        bin_size = in_bin.sum()
        
        # Weighted average
        ece += (bin_size / len(y_pred)) * np.abs(bin_accuracy - bin_confidence)
    
    return ece


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    y_pred_scores: np.ndarray,
    threshold: float = 0.5,
    k_values: list = [10, 20, 50]
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities (for classification)
        y_pred_scores: Predicted scores (for ranking)
        threshold: Classification threshold
        k_values: K values for Recall@K
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Classification metrics
    metrics['auroc'] = compute_auroc(y_true, y_pred_proba)
    metrics['auprc'] = compute_auprc(y_true, y_pred_proba)
    
    prf = compute_precision_recall_f1(y_true, y_pred_proba, threshold)
    metrics.update(prf)
    
    # Ranking metrics
    for k in k_values:
        metrics[f'recall@{k}'] = compute_recall_at_k(y_true, y_pred_scores, k)
    
    metrics['ndcg'] = compute_ndcg(y_true, y_pred_scores)
    
    # Calibration
    metrics['ece'] = compute_ece(y_true, y_pred_proba)
    
    return metrics


# Import guard for optional dependency
from typing import Optional

__all__ = [
    'compute_auroc',
    'compute_auprc', 
    'compute_precision_recall_f1',
    'compute_recall_at_k',
    'compute_ndcg',
    'compute_ece',
    'compute_all_metrics'
]
