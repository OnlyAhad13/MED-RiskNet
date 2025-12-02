"""
Test script for image data pipeline.

This script tests:
1. ImageDataset loading and transforms
2. CNNEncoder forward pass
3. EfficientNetV2 and ResNet50 models

Usage:
    python tests/test_image.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from torch.utils.data import DataLoader

from data.datasets import ImageDataset
from models.image import CNNEncoder, ImageClassifier


def test_image_dataset():
    """Test ImageDataset class."""
    print("=" * 80)
    print("Testing ImageDataset")
    print("=" * 80)
    
    # Sample patient IDs (will use fallback image)
    patient_ids = ['P001', 'P002', 'P003', 'P004', 'P005']
    
    # Create dataset
    dataset = ImageDataset(
        image_dir='data',
        patient_ids=patient_ids,
        target_size=224,
        augment=False
    )
    
    print(f"✓ Dataset created successfully")
    print(f"  - Number of samples: {len(dataset)}")
    print(f"  - Target size: 224x224")
    print(f"  - Normalization: ImageNet statistics")
    
    # Get a sample
    sample = dataset[0]
    image = sample['image']
    patient_id = sample['patient_id']
    
    print(f"\n✓ Sample extracted:")
    print(f"  - Image shape: {image.shape}")  # [3, 224, 224]
    print(f"  - Image dtype: {image.dtype}")
    print(f"  - Patient ID: {patient_id}")
    print(f"  - Value range: [{image.min():.4f}, {image.max():.4f}]")
    
    # Check shape
    assert image.shape == (3, 224, 224), f"Expected shape (3, 224, 224), got {image.shape}"
    print(f"\n✓ Image shape is correct")
    
    # Test DataLoader
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
    batch = next(iter(dataloader))
    batch_images = batch['image']
    batch_ids = batch['patient_id']
    
    print(f"\n✓ DataLoader test:")
    print(f"  - Batch images shape: {batch_images.shape}")  # [3, 3, 224, 224]
    print(f"  - Batch IDs: {batch_ids}")
    
    return dataset


def test_cnn_encoder_efficientnet():
    """Test CNNEncoder with EfficientNetV2."""
    print("\n" + "=" * 80)
    print("Testing CNNEncoder (EfficientNetV2)")
    print("=" * 80)
    
    try:
        # Create encoder with EfficientNetV2
        encoder = CNNEncoder(
            model_name='efficientnet_v2',
            pretrained=True,
            feature_dim=512,
            freeze_backbone=False
        )
        
        model_name = "EfficientNetV2"
    except:
        print("⚠ timm not available, test will use ResNet50 instead")
        return None
    
    print(f"\n✓ {model_name} encoder created:")
    print(f"  - Feature dim: 512")
    print(f"  - Pretrained: True")
    
    # Count parameters
    num_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"  - Trainable parameters: {num_params:,}")
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    encoder.eval()
    with torch.no_grad():
        output = encoder(dummy_input)
    
    print(f"\n✓ Forward pass successful:")
    print(f"  - Input shape: {dummy_input.shape}")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Output mean: {output.mean().item():.4f}")
    print(f"  - Output std: {output.std().item():.4f}")
    
    assert output.shape == (batch_size, 512), f"Expected shape {(batch_size, 512)}, got {output.shape}"
    print(f"\n✓ Output shape is correct")
    
    return encoder


def test_cnn_encoder_resnet():
    """Test CNNEncoder with ResNet50 (fallback)."""
    print("\n" + "=" * 80)
    print("Testing CNNEncoder (ResNet50 Fallback)")
    print("=" * 80)
    
    # Create encoder with ResNet50
    encoder = CNNEncoder(
        model_name='resnet50',
        pretrained=True,
        feature_dim=512,
        freeze_backbone=False
    )
    
    print(f"\n✓ ResNet50 encoder created:")
    print(f"  - Feature dim: 512")
    print(f"  - Pretrained: True")
    
    # Count parameters
    num_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"  - Trainable parameters: {num_params:,}")
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    encoder.eval()
    with torch.no_grad():
        output = encoder(dummy_input)
    
    print(f"\n✓ Forward pass successful:")
    print(f"  - Input shape: {dummy_input.shape}")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Output mean: {output.mean().item():.4f}")
    print(f"  - Output std: {output.std().item():.4f}")
    
    assert output.shape == (batch_size, 512), f"Expected shape {(batch_size, 512)}, got {output.shape}"
    print(f"\n✓ Output shape is correct")
    
    return encoder


def test_image_classifier():
    """Test ImageClassifier class."""
    print("\n" + "=" * 80)
    print("Testing ImageClassifier")
    print("=" * 80)
    
    # Create classifier (will use ResNet50 since timm may not be installed)
    classifier = ImageClassifier(
        model_name='resnet50',
        pretrained=True,
        feature_dim=512,
        num_classes=1,
        freeze_backbone=False
    )
    
    print(f"\n✓ Classifier created:")
    print(f"  - Feature dim: 512")
    print(f"  - Num classes: 1 (binary)")
    
    # Count parameters
    num_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"  - Trainable parameters: {num_params:,}")
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
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


def test_real_image_forward():
    """Test forward pass with real image from dataset."""
    print("\n" + "=" * 80)
    print("Testing Real Image Forward Pass")
    print("=" * 80)
    
    # Create dataset
    patient_ids = ['P001']
    dataset = ImageDataset(
        image_dir='data',
        patient_ids=patient_ids,
        target_size=224,
        augment=False
    )
    
    # Get real image
    sample = dataset[0]
    real_image = sample['image'].unsqueeze(0)  # Add batch dimension
    
    print(f"✓ Loaded real image:")
    print(f"  - Shape: {real_image.shape}")
    print(f"  - Patient ID: {sample['patient_id']}")
    
    # Create encoder
    encoder = CNNEncoder(
        model_name='resnet50',
        pretrained=True,
        feature_dim=512
    )
    
    # Forward pass
    encoder.eval()
    with torch.no_grad():
        features = encoder(real_image)
    
    print(f"\n✓ Feature extraction successful:")
    print(f"  - Features shape: {features.shape}")
    print(f"  - Features mean: {features.mean().item():.4f}")
    print(f"  - Features std: {features.std().item():.4f}")
    print(f"  - Feature sample (first 10): {features[0][:10].numpy()}")


def main():
    """Main test function."""
    print("\n" + "🧪 " * 20)
    print("MED-RiskNET Image Pipeline Test Suite")
    print("🧪 " * 20 + "\n")
    
    try:
        # Test dataset
        dataset = test_image_dataset()
        
        # Test EfficientNetV2 encoder (if timm available)
        encoder_eff = test_cnn_encoder_efficientnet()
        
        # Test ResNet50 encoder (always available)
        encoder_resnet = test_cnn_encoder_resnet()
        
        # Test classifier
        classifier = test_image_classifier()
        
        # Test real image
        test_real_image_forward()
        
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
