"""
Evaluation script for MED-RiskNET.

Loads trained model and computes comprehensive evaluation metrics.

Usage:
    python evaluation/evaluate.py --checkpoint path/to/checkpoint.pt
"""

import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse

from models.medrisknet import MedRiskNet
from data.multimodal_dataset import MultimodalDataset, collate_multimodal
from evaluation.metrics import compute_all_metrics
from evaluation.calibration import calibrate_model, TemperatureScaling
from evaluation.uncertainty import MCDropoutWrapper
from training.trainer import set_seed


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str = 'cpu'
) -> MedRiskNet:
    """Load model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = MedRiskNet(
        tabular_input_dim=8,
        num_classes=1,
        fusion_hidden_dim=128,
        use_graph=True
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded from epoch {checkpoint['epoch']}")
    
    return model


def evaluate_model(
    model: MedRiskNet,
    dataloader: DataLoader,
    device: str = 'cpu',
    use_calibration: bool = True,
    use_uncertainty: bool = True
) -> dict:
    """
    Evaluate model on dataset.
    
    Args:
        model: Trained model
        dataloader: Data loader
        device: Device
        use_calibration: Whether to apply temperature scaling
        use_uncertainty: Whether to compute MC-Dropout uncertainty
    
    Returns:
        Dictionary with evaluation results
    """
    model.eval()
    
    # Collect predictions
    all_probs = []
    all_scores = []
    all_labels = []
    all_uncertainties = []
    
    print(f"\nRunning evaluation...")
    
    # Optional: Temperature calibration
    temp_scaling = None
    if use_calibration:
        print("Fitting temperature scaling...")
        temp_scaling = calibrate_model(model, dataloader, device)
    
    # Optional: MC-Dropout wrapper
    mc_model = None
    if use_uncertainty:
        print("Initializing MC-Dropout...")
        mc_model = MCDropoutWrapper(model, num_samples=20)
    
    # Evaluate
    with torch.no_grad():
        for batch in dataloader:
            tabular = batch['tabular'].to(device)
            labels = batch['label'].to(device)
            
            # Optional modalities
            image = batch.get('image')
            if image is not None:
                image = image.to(device)
            text = batch.get('text')
            
            # Build graph
            graph_x = None
            edge_index = None
            if model.use_graph:
                from data.graph_builder import PatientGraphBuilder
                builder = PatientGraphBuilder(k_neighbors=3)
                edge_index = builder.build_graph_from_features(tabular)
                graph_x = tabular
                edge_index = edge_index.to(device)
            
            # Forward pass
            outputs = model(
                tabular=tabular,
                image=image,
                text=text,
                graph_x=graph_x,
                edge_index=edge_index
            )
            
            probs = outputs['classification'].cpu().numpy()
            scores = outputs['ranking'].cpu().numpy()
            
            # Apply calibration if available
            if temp_scaling is not None:
                logits = torch.logit(
                    outputs['classification'].clamp(min=1e-7, max=1-1e-7)
                )
                calibrated_probs = temp_scaling.calibrate_probabilities(logits)
                probs = calibrated_probs.cpu().numpy()
            
            # Compute uncertainty if enabled
            if mc_model is not None:
                unc_outputs = mc_model.forward_with_uncertainty(
                    tabular=tabular,
                    image=image,
                    text=text,
                    graph_x=graph_x,
                    edge_index=edge_index
                )
                uncertainties = unc_outputs['std_classification'].cpu().numpy()
                all_uncertainties.append(uncertainties)
            
            all_probs.append(probs)
            all_scores.append(scores)
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate all batches
    all_probs = np.concatenate(all_probs, axis=0).squeeze()
    all_scores = np.concatenate(all_scores, axis=0).squeeze()
    all_labels = np.concatenate(all_labels, axis=0)
    
    if use_uncertainty:
        all_uncertainties = np.concatenate(all_uncertainties, axis=0).squeeze()
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_all_metrics(
        y_true=all_labels,
        y_pred_proba=all_probs,
        y_pred_scores=all_scores,
        threshold=0.5,
        k_values=[3, 5, 10]
    )
    
    # Add uncertainty metrics if available
    if use_uncertainty:
        metrics['mean_uncertainty'] = all_uncertainties.mean()
        metrics['max_uncertainty'] = all_uncertainties.max()
    
    return metrics


def print_evaluation_results(metrics: dict):
    """Print evaluation results in a nice format."""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    print("\nClassification Metrics:")
    print(f"  - AUROC:     {metrics['auroc']:.4f}")
    print(f"  - AUPRC:     {metrics['auprc']:.4f}")
    print(f"  - Precision: {metrics['precision']:.4f}")
    print(f"  - Recall:    {metrics['recall']:.4f}")
    print(f"  - F1 Score:  {metrics['f1']:.4f}")
    
    print("\nRanking Metrics:")
    print(f"  - Recall@3:  {metrics['recall@3']:.4f}")
    print(f"  - Recall@5:  {metrics['recall@5']:.4f}")
    print(f"  - Recall@10: {metrics['recall@10']:.4f}")
    print(f"  - NDCG:      {metrics['ndcg']:.4f}")
    
    print("\nCalibration:")
    print(f"  - ECE:       {metrics['ece']:.4f}")
    
    if 'mean_uncertainty' in metrics:
        print("\nUncertainty:")
        print(f"  - Mean:      {metrics['mean_uncertainty']:.4f}")
        print(f"  - Max:       {metrics['max_uncertainty']:.4f}")
    
    print("\n" + "=" * 80)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate MED-RiskNET')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='test_checkpoints/best_model.pt',
        help='Path to checkpoint'
    )
    parser.add_argument(
        '--data-csv',
        type=str,
        default='data/sample_patient_data.csv',
        help='Path to data CSV'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        default='data',
        help='Path to image directory'
    )
    parser.add_argument(
        '--text-file',
        type=str,
        default='data/sample_medical_reports.txt',
        help='Path to text file'
    )
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-calibration', action='store_true', help='Disable calibration')
    parser.add_argument('--no-uncertainty', action='store_true', help='Disable uncertainty')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    print("\n" + "🧪 " * 20)
    print("MED-RiskNET Model Evaluation")
    print("🧪 " * 20)
    
    # Load model
    model = load_model_from_checkpoint(args.checkpoint, args.device)
    
    # Load dataset
    print(f"\nLoading dataset...")
    dataset = MultimodalDataset(
        csv_path=args.data_csv,
        image_dir=args.image_dir,
        text_file=args.text_file
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_multimodal
    )
    
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    
    # Evaluate
    metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=args.device,
        use_calibration=not args.no_calibration,
        use_uncertainty=not args.no_uncertainty
    )
    
    # Print results
    print_evaluation_results(metrics)


if __name__ == '__main__':
    main()
