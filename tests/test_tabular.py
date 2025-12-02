"""
Test script for tabular data pipeline.

This script tests:
1. TabularDataset loading and preprocessing
2. TabularEncoder forward pass
3. Complete pipeline integration

Usage:
    python tests/test_tabular.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from data.datasets import TabularDataset
from models.tabular import TabularEncoder, TabularClassifier, ResidualTabularEncoder


def test_dataset():
    """Test TabularDataset class."""
    print("=" * 80)
    print("Testing TabularDataset")
    print("=" * 80)
    
    # Define features
    numeric_features = ['age', 'bmi', 'bp', 'chol', 'glucose']
    categorical_features = ['sex']
    target_column = 'label'
    
    # Create dataset
    dataset = TabularDataset(
        csv_path='data/sample_patient_data.csv',
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        target_column=target_column,
        is_train=True
    )
    
    print(f"✓ Dataset loaded successfully")
    print(f"  - Number of samples: {len(dataset)}")
    print(f"  - Feature dimension: {dataset.feature_dim}")
    print(f"  - Numeric features: {len(numeric_features)}")
    print(f"  - Categorical features: {len(categorical_features)}")
    
    # Get a sample
    sample = dataset[0]
    features = sample['features']
    target = sample['target']
    print(f"\n✓ Sample extracted:")
    print(f"  - Features shape: {features.shape}")
    print(f"  - Features dtype: {features.dtype}")
    print(f"  - Target: {target.item()}")
    print(f"  - Features (first 10): {features[:10].numpy()}")
    
    # Check for NaN values
    assert not torch.isnan(features).any(), "Features contain NaN values"
    print(f"\n✓ No NaN values in features")
    
    # Test DataLoader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    batch = next(iter(dataloader))
    batch_features = batch['features']
    batch_targets = batch['target']
    print(f"\n✓ DataLoader test:")
    print(f"  - Batch features shape: {batch_features.shape}")
    print(f"  - Batch targets shape: {batch_targets.shape}")
    
    return dataset


def test_encoder(input_dim):
    """Test TabularEncoder class."""
    print("\n" + "=" * 80)
    print("Testing TabularEncoder")
    print("=" * 80)
    
    # Create encoder with default settings
    encoder = TabularEncoder(
        input_dim=input_dim,
        hidden_dims=[256, 128, 64],
        output_dim=32,
        dropout=0.3,
        activation='relu'
    )
    
    print(f"✓ Encoder created:")
    print(f"  - Input dim: {input_dim}")
    print(f"  - Hidden dims: [256, 128, 64]")
    print(f"  - Output dim: 32")
    print(f"  - Dropout: 0.3")
    
    # Count parameters
    num_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"  - Trainable parameters: {num_params:,}")
    
    # Test forward pass
    batch_size = 16
    dummy_input = torch.randn(batch_size, input_dim)
    
    encoder.eval()
    with torch.no_grad():
        output = encoder(dummy_input)
    
    print(f"\n✓ Forward pass successful:")
    print(f"  - Input shape: {dummy_input.shape}")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Output mean: {output.mean().item():.4f}")
    print(f"  - Output std: {output.std().item():.4f}")
    
    assert output.shape == (batch_size, 32), f"Expected output shape {(batch_size, 32)}, got {output.shape}"
    print(f"\n✓ Output shape is correct")
    
    return encoder


def test_classifier(input_dim):
    """Test TabularClassifier class."""
    print("\n" + "=" * 80)
    print("Testing TabularClassifier")
    print("=" * 80)
    
    # Create classifier
    classifier = TabularClassifier(
        input_dim=input_dim,
        hidden_dims=[128, 64],
        num_classes=1,
        dropout=0.2,
        activation='relu'
    )
    
    print(f"✓ Classifier created:")
    print(f"  - Input dim: {input_dim}")
    print(f"  - Hidden dims: [128, 64]")
    print(f"  - Num classes: 1 (binary)")
    
    # Count parameters
    num_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"  - Trainable parameters: {num_params:,}")
    
    # Test forward pass
    batch_size = 16
    dummy_input = torch.randn(batch_size, input_dim)
    
    classifier.eval()
    with torch.no_grad():
        predictions = classifier(dummy_input)
    
    print(f"\n✓ Forward pass successful:")
    print(f"  - Input shape: {dummy_input.shape}")
    print(f"  - Predictions shape: {predictions.shape}")
    print(f"  - Predictions range: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")
    
    # Test with embedding return
    with torch.no_grad():
        predictions, embeddings = classifier(dummy_input, return_embedding=True)
    
    print(f"\n✓ Forward pass with embeddings:")
    print(f"  - Predictions shape: {predictions.shape}")
    print(f"  - Embeddings shape: {embeddings.shape}")
    
    return classifier


def test_residual_encoder(input_dim):
    """Test ResidualTabularEncoder class."""
    print("\n" + "=" * 80)
    print("Testing ResidualTabularEncoder")
    print("=" * 80)
    
    # Create residual encoder
    res_encoder = ResidualTabularEncoder(
        input_dim=input_dim,
        hidden_dim=256,
        num_blocks=3,
        output_dim=64,
        dropout=0.3
    )
    
    print(f"✓ Residual encoder created:")
    print(f"  - Input dim: {input_dim}")
    print(f"  - Hidden dim: 256")
    print(f"  - Num blocks: 3")
    print(f"  - Output dim: 64")
    
    # Count parameters
    num_params = sum(p.numel() for p in res_encoder.parameters() if p.requires_grad)
    print(f"  - Trainable parameters: {num_params:,}")
    
    # Test forward pass
    batch_size = 16
    dummy_input = torch.randn(batch_size, input_dim)
    
    res_encoder.eval()
    with torch.no_grad():
        output = res_encoder(dummy_input)
    
    print(f"\n✓ Forward pass successful:")
    print(f"  - Input shape: {dummy_input.shape}")
    print(f"  - Output shape: {output.shape}")
    
    return res_encoder


def test_training_step(dataset, encoder):
    """Test a simple training step."""
    print("\n" + "=" * 80)
    print("Testing Training Step")
    print("=" * 80)
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Add a simple classification head
    # Get embedding dimension from encoder's last layer output_dim (32)
    classifier_head = torch.nn.Linear(32, 1)
    
    # Loss and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(classifier_head.parameters()),
        lr=0.001
    )
    
    # Training mode
    encoder.train()
    classifier_head.train()
    
    # Get a batch
    batch = next(iter(dataloader))
    features = batch['features']
    targets = batch['target'].float()  # Convert to float for BCEWithLogitsLoss
    
    print(f"✓ Batch loaded:")
    print(f"  - Features shape: {features.shape}")
    print(f"  - Targets shape: {targets.shape}")
    
    # Forward pass
    embeddings = encoder(features)
    logits = classifier_head(embeddings).squeeze()
    loss = criterion(logits, targets)
    
    print(f"\n✓ Forward pass completed:")
    print(f"  - Embeddings shape: {embeddings.shape}")
    print(f"  - Logits shape: {logits.shape}")
    print(f"  - Loss: {loss.item():.4f}")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"\n✓ Backward pass completed")
    print(f"  - Gradients computed and parameters updated")
    
    # Verify gradients
    has_grads = any(p.grad is not None for p in encoder.parameters())
    assert has_grads, "No gradients computed"
    print(f"  - Gradients verified ✓")


def test_inference(dataset, classifier):
    """Test inference mode."""
    print("\n" + "=" * 80)
    print("Testing Inference Mode")
    print("=" * 80)
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    
    # Get all data
    batch = next(iter(dataloader))
    all_features = batch['features']
    all_targets = batch['target']
    
    # Inference mode
    classifier.eval()
    with torch.no_grad():
        predictions = classifier(all_features)
    
    predictions = predictions.squeeze().numpy()
    targets = all_targets.numpy()
    
    print(f"✓ Inference completed:")
    print(f"  - Total samples: {len(predictions)}")
    print(f"  - Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    
    # Simple metrics
    predicted_labels = (predictions > 0.5).astype(int)
    accuracy = (predicted_labels == targets).mean()
    
    print(f"\n✓ Simple metrics:")
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - Positive predictions: {predicted_labels.sum()} / {len(predicted_labels)}")
    print(f"  - Actual positives: {targets.sum()} / {len(targets)}")


def main():
    """Main test function."""
    print("\n" + "🧪 " * 20)
    print("MED-RiskNet Tabular Pipeline Test Suite")
    print("🧪 " * 20 + "\n")
    
    try:
        # Test dataset
        dataset = test_dataset()
        input_dim = dataset.feature_dim
        
        # Test encoder
        encoder = test_encoder(input_dim)
        
        # Test classifier
        classifier = test_classifier(input_dim)
        
        # Test residual encoder
        res_encoder = test_residual_encoder(input_dim)
        
        # Test training step
        test_training_step(dataset, encoder)
        
        # Test inference
        test_inference(dataset, classifier)
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"TEST FAILED: {str(e)}")
        print("=" * 80 + "\n")
        raise


if __name__ == '__main__':
    main()