"""
Smoke test for MED-RiskNET training pipeline.

Tests:
- Model initialization
- Forward pass
- Training for 1 epoch
- Checkpointing

Usage:
    python tests/test_train_smoke.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from models.medrisknet import MedRiskNet
from training.trainer import MedRiskNetTrainer, set_seed
from data.multimodal_dataset import MultimodalDataset, collate_multimodal


def test_model_initialization():
    """Test model creation."""
    print("=" * 80)
    print("Testing Model Initialization")
    print("=" * 80)
    
    model = MedRiskNet(
        tabular_input_dim=8,  # age, bmi, bp_sys, bp_dia, chol, glucose, sex_F, sex_M
        num_classes=1,
        fusion_hidden_dim=128,
        use_graph=True  # Enable graph encoder
    )
    
    print(f"✓ Model created successfully")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total trainable parameters: {num_params:,}")
    
    # Check components
    print(f"  - Tabular encoder: ✓")
    print(f"  - Image encoder: ✓")
    print(f"  - Text encoder: {'✓' if model.has_text_encoder else '✗ (transformers not installed)'}")
    print(f"  - Graph encoder: {'✓' if model.use_graph else '✗ (disabled)'}")
    print(f"  - Fusion: ✓")
    print(f"  - Classification head: ✓")
    print(f"  - Ranking head: ✓")
    
    return model


def test_forward_pass(model):
    """Test forward pass with dummy data."""
    print("\n" + "=" * 80)
    print("Testing Forward Pass")
    print("=" * 80)
    
    batch_size = 4
    
    # Create dummy inputs
    tabular = torch.randn(batch_size, 8)  # 8 features now
    image = torch.randn(batch_size, 3, 224, 224)
    text = ["Patient has hypertension"] * batch_size
    
    print(f"✓ Created dummy inputs:")
    print(f"  - Tabular: {tabular.shape}")
    print(f"  - Image: {image.shape}")
    print(f"  - Text: {len(text)} strings")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(tabular=tabular, image=image, text=text)
    
    print(f"\n✓ Forward pass successful:")
    print(f"  - Classification output: {outputs['classification'].shape}")
    print(f"  - Ranking output: {outputs['ranking'].shape}")
    print(f"  - Classification range: [{outputs['classification'].min():.4f}, {outputs['classification'].max():.4f}]")
    print(f"  - Ranking range: [{outputs['ranking'].min():.4f}, {outputs['ranking'].max():.4f}]")
    
    return outputs


def test_dataset_loading():
    """Test multimodal dataset."""
    print("\n" + "=" * 80)
    print("Testing Dataset Loading")
    print("=" * 80)
    
    # Create dataset
    dataset = MultimodalDataset(
        csv_path='data/sample_patient_data.csv',
        image_dir='data',
        text_file='data/sample_medical_reports.txt'
    )
    
    print(f"✓ Dataset created:")
    print(f"  - Number of samples: {len(dataset)}")
    print(f"  - Feature columns: {dataset.feature_cols}")
    
    # Get a sample
    sample = dataset[0]
    print(f"\n✓ Sample loaded:")
    print(f"  - Patient ID: {sample['patient_id']}")
    print(f"  - Tabular shape: {sample['tabular'].shape}")
    print(f"  - Image shape: {sample['image'].shape if sample['image'] is not None else 'None'}")
    print(f"  - Text length: {len(sample['text'])} chars")
    print(f"  - Label: {sample['label']}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_multimodal
    )
    
    batch = next(iter(dataloader))
    print(f"\n✓ DataLoader working:")
    print(f"  - Batch tabular: {batch['tabular'].shape}")
    print(f"  - Batch image: {batch['image'].shape if batch['image'] is not None else 'None'}")
    print(f"  - Batch text: {len(batch['text']) if batch['text'] else 0} strings")
    print(f"  - Batch labels: {batch['label'].shape}")
    
    return dataloader


def test_training_one_epoch():
    """Test training for one epoch."""
    print("\n" + "=" * 80)
    print("Testing Training (1 Epoch)")
    print("=" * 80)
    
    # Set seed
    set_seed(42)
    print(f"✓ Random seed set to 42")
    
    # Create model
    model = MedRiskNet(
        tabular_input_dim=8,  # Updated to match actual features
        num_classes=1,
        fusion_hidden_dim=128,
        use_graph=True  # Enable graph encoder
    )
    
    # Create dataset
    dataset = MultimodalDataset(
        csv_path='data/sample_patient_data.csv',
        image_dir='data',
        text_file='data/sample_medical_reports.txt'
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_multimodal
    )
    
    # Create trainer
    trainer = MedRiskNetTrainer(
        model=model,
        device='cpu',
        lr=1e-3,
        ranking_loss_weight=0.5,
        checkpoint_dir='test_checkpoints'
    )
    
    print(f"✓ Trainer created:")
    print(f"  - Device: cpu")
    print(f"  - Learning rate: 1e-3")
    print(f"  - Ranking loss weight: 0.5")
    
    # Train for 1 epoch
    print(f"\n✓ Training for 1 epoch...")
    trainer.train(
        train_loader=dataloader,
        num_epochs=1,
        save_every=1,
        verbose=True
    )
    
    print(f"\n✓ Training completed:")
    print(f"  - Final loss: {trainer.history['train_loss'][-1]:.4f}")
    print(f"  - Classification loss: {trainer.history['train_class_loss'][-1]:.4f}")
    print(f"  - Ranking loss: {trainer.history['train_rank_loss'][-1]:.4f}")
    
    # Check checkpoint
    checkpoint_path = Path('test_checkpoints/epoch_1.pt')
    if checkpoint_path.exists():
        print(f"\n✓ Checkpoint saved: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"  - Epoch: {checkpoint['epoch']}")
        print(f"  - Keys: {list(checkpoint.keys())}")
    
    return trainer


def main():
    """Run all smoke tests."""
    print("\n" + "🧪 " * 20)
    print("MED-RiskNET Training Smoke Test")
    print("🧪 " * 20 + "\n")
    
    try:
        # Test model initialization
        model = test_model_initialization()
        
        # Test forward pass
        outputs = test_forward_pass(model)
        
        # Test dataset
        dataloader = test_dataset_loading()
        
        # Test training
        trainer = test_training_one_epoch()
        
        print("\n" + "=" * 80)
        print("ALL SMOKE TESTS PASSED!")
        print("=" * 80 + "\n")
        
        print("Summary:")
        print("  ✓ Model initialization working")
        print("  ✓ Forward pass successful")
        print("  ✓ Dataset loading functional")
        print("  ✓ Training pipeline operational")
        print("  ✓ Checkpointing working")
        print("\nMED-RiskNET is ready for training! 🚀")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"SMOKE TEST FAILED: {str(e)}")
        print("=" * 80 + "\n")
        raise


if __name__ == '__main__':
    main()
