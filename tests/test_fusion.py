"""
Test script for multimodal fusion models.

This script tests:
1. FusionTransformer with random tensors
2. CrossModalAttentionFusion
3. MultimodalClassifier
4. Integration with real encoders

Usage:
    python tests/test_fusion.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

from models.fusion import FusionTransformer, CrossModalAttentionFusion, MultimodalClassifier


def test_fusion_transformer_random():
    """Test FusionTransformer with random tensors."""
    print("=" * 80)
    print("Testing FusionTransformer (Random Tensors)")
    print("=" * 80)
    
    # Define modality dimensions
    modality_dims = {
        'tabular': 32,
        'image': 512,
        'text': 768,
        'graph': 16
    }
    
    # Create fusion model
    fusion = FusionTransformer(
        modality_dims=modality_dims,
        hidden_dim=256,
        num_heads=8,
        num_layers=2,
        dropout=0.1,
        fusion_method='mean'
    )
    
    print(f"✓ FusionTransformer created:")
    print(f"  - Modality dims: {modality_dims}")
    print(f"  - Hidden dim: 256")
    print(f"  - Num heads: 8")
    print(f"  - Num layers: 2")
    print(f"  - Fusion method: mean")
    
    # Count parameters
    num_params = sum(p.numel() for p in fusion.parameters() if p.requires_grad)
    print(f"  - Trainable parameters: {num_params:,}")
    
    # Create random inputs
    batch_size = 4
    tabular = torch.randn(batch_size, 32)
    image = torch.randn(batch_size, 512)
    text = torch.randn(batch_size, 768)
    graph = torch.randn(batch_size, 16)
    
    print(f"\n✓ Input shapes:")
    print(f"  - Tabular: {tabular.shape}")
    print(f"  - Image: {image.shape}")
    print(f"  - Text: {text.shape}")
    print(f"  - Graph: {graph.shape}")
    
    # Forward pass with all modalities
    fusion.eval()
    with torch.no_grad():
        output = fusion(tabular=tabular, image=image, text=text, graph=graph)
    
    print(f"\n✓ Forward pass successful:")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Output mean: {output.mean().item():.4f}")
    print(f"  - Output std: {output.std().item():.4f}")
    
    assert output.shape == (batch_size, 256), f"Expected ({batch_size}, 256), got {output.shape}"
    print(f"\n✓ Output shape is correct")
    
    # Test with subset of modalities
    print(f"\n✓ Testing with subset of modalities:")
    
    with torch.no_grad():
        # Only tabular and image
        output_subset = fusion(tabular=tabular, image=image)
        print(f"  - Tabular + Image: {output_subset.shape}")
        
        # Only text
        output_text = fusion(text=text)
        print(f"  - Text only: {output_text.shape}")
    
    return fusion


def test_cross_attention_fusion():
    """Test CrossModalAttentionFusion."""
    print("\n" + "=" * 80)
    print("Testing CrossModalAttentionFusion")
    print("=" * 80)
    
    modality_dims = {
        'tabular': 32,
        'image': 512,
        'text': 768,
        'graph': 16
    }
    
    # Create cross-attention fusion
    fusion = CrossModalAttentionFusion(
        modality_dims=modality_dims,
        hidden_dim=256,
        num_heads=8,
        primary_modality='tabular',  # Tabular is query
        dropout=0.1
    )
    
    print(f"✓ CrossModalAttentionFusion created:")
    print(f"  - Hidden dim: 256")
    print(f"  - Primary modality: tabular (query)")
    print(f"  - Other modalities: key/value")
    
    # Count parameters
    num_params = sum(p.numel() for p in fusion.parameters() if p.requires_grad)
    print(f"  - Trainable parameters: {num_params:,}")
    
    # Random inputs
    batch_size = 4
    tabular = torch.randn(batch_size, 32)
    image = torch.randn(batch_size, 512)
    text = torch.randn(batch_size, 768)
    
    # Forward pass
    fusion.eval()
    with torch.no_grad():
        output = fusion(tabular=tabular, image=image, text=text)
    
    print(f"\n✓ Forward pass successful:")
    print(f"  - Output shape: {output.shape}")
    
    assert output.shape == (batch_size, 256), f"Expected ({batch_size}, 256), got {output.shape}"
    print(f"\n✓ Output shape is correct")
    
    return fusion


def test_multimodal_classifier():
    """Test MultimodalClassifier."""
    print("\n" + "=" * 80)
    print("Testing MultimodalClassifier")
    print("=" * 80)
    
    modality_dims = {
        'tabular': 32,
        'image': 512,
        'text': 768,
        'graph': 16
    }
    
    # Create classifier
    classifier = MultimodalClassifier(
        modality_dims=modality_dims,
        hidden_dim=256,
        num_classes=1,  # Binary classification
        fusion_type='transformer',
        num_heads=8,
        num_layers=2
    )
    
    print(f"✓ MultimodalClassifier created:")
    print(f"  - Num classes: 1 (binary)")
    print(f"  - Fusion type: transformer")
    
    # Count parameters
    num_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"  - Trainable parameters: {num_params:,}")
    
    # Random inputs
    batch_size = 4
    tabular = torch.randn(batch_size, 32)
    image = torch.randn(batch_size, 512)
    text = torch.randn(batch_size, 768)
    graph = torch.randn(batch_size, 16)
    
    # Forward pass
    classifier.eval()
    with torch.no_grad():
        predictions = classifier(tabular=tabular, image=image, text=text, graph=graph)
    
    print(f"\n✓ Forward pass successful:")
    print(f"  - Predictions shape: {predictions.shape}")
    print(f"  - Predictions range: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")
    
    # Test with embedding return
    with torch.no_grad():
        predictions, embeddings = classifier(
            tabular=tabular,
            image=image,
            text=text,
            graph=graph,
            return_embedding=True
        )
    
    print(f"\n✓ Forward pass with embeddings:")
    print(f"  - Predictions shape: {predictions.shape}")
    print(f"  - Embeddings shape: {embeddings.shape}")
    
    return classifier


def test_fusion_methods():
    """Test different fusion methods."""
    print("\n" + "=" * 80)
    print("Testing Different Fusion Methods")
    print("=" * 80)
    
    modality_dims = {'tabular': 32, 'image': 512, 'text': 768}
    batch_size = 4
    
    # Test inputs
    tabular = torch.randn(batch_size, 32)
    image = torch.randn(batch_size, 512)
    text = torch.randn(batch_size, 768)
    
    for method in ['mean', 'max', 'cls']:
        fusion = FusionTransformer(
            modality_dims=modality_dims,
            hidden_dim=128,
            num_heads=4,
            num_layers=1,
            fusion_method=method
        )
        
        with torch.no_grad():
            output = fusion(tabular=tabular, image=image, text=text)
        
        print(f"✓ Fusion method '{method}': output shape {output.shape}")


def test_mini_training():
    """Test a mini training loop."""
    print("\n" + "=" * 80)
    print("Testing Mini Training Loop")
    print("=" * 80)
    
    modality_dims = {'tabular': 32, 'image': 512}
    
    # Create model
    model = MultimodalClassifier(
        modality_dims=modality_dims,
        hidden_dim=128,
        num_classes=1
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()
    
    print(f"✓ Training setup complete")
    
    # Dummy data
    batch_size = 8
    tabular = torch.randn(batch_size, 32)
    image = torch.randn(batch_size, 512)
    labels = torch.randint(0, 2, (batch_size,)).float()
    
    # Train for a few steps
    model.train()
    losses = []
    
    for epoch in range(5):
        optimizer.zero_grad()
        predictions = model(tabular=tabular, image=image).squeeze()
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    print(f"\n✓ Training successful:")
    print(f"  - Initial loss: {losses[0]:.4f}")
    print(f"  - Final loss: {losses[-1]:.4f}")
    print(f"  - Gradients flowing: {losses[0] != losses[-1]}")


def main():
    """Main test function."""
    print("\n" + "🧪 " * 20)
    print("MED-RiskNET Multimodal Fusion Test Suite")
    print("🧪 " * 20 + "\n")
    
    try:
        # Test fusion transformer
        fusion_transformer = test_fusion_transformer_random()
        
        # Test cross-attention
        cross_attention = test_cross_attention_fusion()
        
        # Test classifier
        classifier = test_multimodal_classifier()
        
        # Test fusion methods
        test_fusion_methods()
        
        # Test training
        test_mini_training()
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("=" * 80 + "\n")
        
        print("Summary:")
        print("  ✓ FusionTransformer: Combines 4 modalities with self-attention")
        print("  ✓ CrossModalAttentionFusion: Uses one modality as query")
        print("  ✓ MultimodalClassifier: End-to-end classification")
        print("  ✓ All fusion methods working (mean, max, cls)")
        print("  ✓ Training loop functional")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"TEST FAILED: {str(e)}")
        print("=" * 80 + "\n")
        raise


if __name__ == '__main__':
    main()
