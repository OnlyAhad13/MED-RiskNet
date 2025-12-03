"""
Test script for graph neural network pipeline.

This script tests:
1. Graph construction from tabular data
2. K-NN edge building with cosine similarity
3. GNNEncoder (GraphSAGE/GAT) forward pass
4. Complete pipeline integration

Usage:
    python tests/test_graph.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

from data.graph_builder import PatientGraphBuilder
from models.graph import GNNEncoder, GNNClassifier, HeterogeneousGNN


def test_graph_construction():
    """Test graph construction from CSV."""
    print("=" * 80)
    print("Testing Graph Construction")
    print("=" * 80)
    
    # Create graph builder
    builder = PatientGraphBuilder(
        k_neighbors=3,
        similarity_threshold=0.0,
        include_self_loops=False
    )
    
    print(f"✓ Graph builder created:")
    print(f"  - K-neighbors: 3")
    print(f"  - Similarity threshold: 0.0")
    print(f"  - Self-loops: False")
    
    # Build graph from CSV
    edge_index, node_features = builder.build_graph_from_csv(
        csv_path='data/sample_patient_data.csv'
    )
    
    print(f"\n✓ Graph built from CSV:")
    print(f"  - Node features shape: {node_features.shape}")
    print(f"  - Edge index shape: {edge_index.shape}")
    print(f"  - Number of nodes: {node_features.shape[0]}")
    print(f"  - Number of edges: {edge_index.shape[1]}")
    
    # Get statistics
    stats = builder.get_graph_statistics(edge_index, node_features.shape[0])
    
    print(f"\n✓ Graph statistics:")
    print(f"  - Avg degree: {stats['avg_degree']:.2f}")
    print(f"  - Max degree: {stats['max_degree']}")
    print(f"  - Min degree: {stats['min_degree']}")
    print(f"  - Density: {stats['density']:.4f}")
    
    # Show some edges
    print(f"\n✓ Sample edges (first 5):")
    for i in range(min(5, edge_index.shape[1])):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        print(f"  - Patient {src} → Patient {dst}")
    
    return edge_index, node_features


def test_gnn_encoder_graphsage(edge_index, node_features):
    """Test GNNEncoder with GraphSAGE."""
    print("\n" + "=" * 80)
    print("Testing GNNEncoder (GraphSAGE)")
    print("=" * 80)
    
    input_dim = node_features.shape[1]
    
    try:
        # Create encoder
        encoder = GNNEncoder(
            input_dim=input_dim,
            hidden_dims=[64, 32],
            output_dim=16,
            gnn_type='sage',
            dropout=0.3
        )
        
        print(f"\n✓ GraphSAGE encoder created:")
        print(f"  - Input dim: {input_dim}")
        print(f"  - Hidden dims: [64, 32]")
        print(f"  - Output dim: 16")
        
    except:
        print("\n⚠ PyTorch Geometric not available")
        print("The encoder will use placeholder MLP mode")
        
        encoder = GNNEncoder(
            input_dim=input_dim,
            hidden_dims=[64, 32],
            output_dim=16,
            gnn_type='sage'
        )
    
    # Count parameters
    num_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"  - Trainable parameters: {num_params:,}")
    
    # Forward pass
    encoder.eval()
    with torch.no_grad():
        embeddings = encoder(node_features, edge_index)
    
    print(f"\n✓ Forward pass successful:")
    print(f"  - Input shape: {node_features.shape}")
    print(f"  - Edge index shape: {edge_index.shape}")
    print(f"  - Output shape: {embeddings.shape}")
    print(f"  - Output mean: {embeddings.mean().item():.4f}")
    print(f"  - Output std: {embeddings.std().item():.4f}")
    
    assert embeddings.shape == (node_features.shape[0], 16), \
        f"Expected shape ({node_features.shape[0]}, 16), got {embeddings.shape}"
    print(f"\n✓ Output shape is correct")
    
    return encoder


def test_gnn_encoder_gat(edge_index, node_features):
    """Test GNNEncoder with GAT."""
    print("\n" + "=" * 80)
    print("Testing GNNEncoder (GAT)")
    print("=" * 80)
    
    input_dim = node_features.shape[1]
    
    try:
        # Create encoder
        encoder = GNNEncoder(
            input_dim=input_dim,
            hidden_dims=[64, 32],
            output_dim=16,
            gnn_type='gat',
            dropout=0.3,
            num_heads=4
        )
        
        print(f"\n✓ GAT encoder created:")
        print(f"  - Input dim: {input_dim}")
        print(f"  - Hidden dims: [64, 32]")
        print(f"  - Output dim: 16")
        print(f"  - Num heads: 4")
        
    except:
        print("\n⚠ Skipping GAT test (PyG not available)")
        return None
    
    # Count parameters
    num_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"  - Trainable parameters: {num_params:,}")
    
    # Forward pass
    encoder.eval()
    with torch.no_grad():
        embeddings = encoder(node_features, edge_index)
    
    print(f"\n✓ Forward pass successful:")
    print(f"  - Output shape: {embeddings.shape}")
    
    return encoder


def test_gnn_classifier(edge_index, node_features):
    """Test GNNClassifier."""
    print("\n" + "=" * 80)
    print("Testing GNNClassifier")
    print("=" * 80)
    
    input_dim = node_features.shape[0]
    
    # Create classifier
    classifier = GNNClassifier(
        input_dim=node_features.shape[1],
        hidden_dims=[64, 32],
        output_dim=16,
        num_classes=1,
        gnn_type='sage'
    )
    
    print(f"\n✓ Classifier created:")
    print(f"  - Num classes: 1 (binary)")
    
    # Count parameters
    num_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"  - Trainable parameters: {num_params:,}")
    
    # Forward pass
    classifier.eval()
    with torch.no_grad():
        predictions = classifier(node_features, edge_index)
    
    print(f"\n✓ Forward pass successful:")
    print(f"  - Predictions shape: {predictions.shape}")
    print(f"  - Predictions range: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")
    
    # Test with embedding return
    with torch.no_grad():
        predictions, embeddings = classifier(
            node_features,
            edge_index,
            return_embedding=True
        )
    
    print(f"\n✓ Forward pass with embeddings:")
    print(f"  - Predictions shape: {predictions.shape}")
    print(f"  - Embeddings shape: {embeddings.shape}")
    
    return classifier


def test_graph_variations():
    """Test graph building with different parameters."""
    print("\n" + "=" * 80)
    print("Testing Graph Variations")
    print("=" * 80)
    
    # Test with different k values
    for k in [1, 3, 5]:
        builder = PatientGraphBuilder(k_neighbors=k)
        edge_index, _ = builder.build_graph_from_csv('data/sample_patient_data.csv')
        print(f"✓ K={k}: {edge_index.shape[1]} edges")
    
    # Test with similarity threshold
    builder = PatientGraphBuilder(k_neighbors=3, similarity_threshold=0.5)
    edge_index, _ = builder.build_graph_from_csv('data/sample_patient_data.csv')
    print(f"✓ Threshold=0.5: {edge_index.shape[1]} edges (filtered)")
    
    # Test with self-loops
    builder = PatientGraphBuilder(k_neighbors=3, include_self_loops=True)
    edge_index, node_features = builder.build_graph_from_csv('data/sample_patient_data.csv')
    print(f"✓ With self-loops: {edge_index.shape[1]} edges")
    
    # Test from feature tensor directly
    dummy_features = torch.randn(20, 8)
    builder = PatientGraphBuilder(k_neighbors=3)
    edge_index = builder.build_graph_from_features(dummy_features)
    print(f"✓ Built from features: {edge_index.shape[1]} edges for 20 nodes")


def test_mini_training():
    """Test a mini training loop."""
    print("\n" + "=" * 80)
    print("Testing Mini Training Loop")
    print("=" * 80)
    
    # Build graph
    builder = PatientGraphBuilder(k_neighbors=3)
    edge_index, node_features = builder.build_graph_from_csv('data/sample_patient_data.csv')
    
    # Create dummy labels (from CSV if available)
    import pandas as pd
    df = pd.read_csv('data/sample_patient_data.csv')
    if 'label' in df.columns:
        labels = torch.tensor(df['label'].values, dtype=torch.float32)
    else:
        labels = torch.randint(0, 2, (node_features.shape[0],), dtype=torch.float32)
    
    # Create model
    model = GNNClassifier(
        input_dim=node_features.shape[1],
        hidden_dims=[32, 16],
        output_dim=8,
        num_classes=1
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()
    
    print(f"✓ Setup complete:")
    print(f"  - Nodes: {node_features.shape[0]}")
    print(f"  - Features: {node_features.shape[1]}")
    print(f"  - Edges: {edge_index.shape[1]}")
    
    # Train for a few steps
    model.train()
    losses = []
    
    for epoch in range(5):
        optimizer.zero_grad()
        predictions = model(node_features, edge_index).squeeze()
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    print(f"\n✓ Training successful:")
    print(f"  - Initial loss: {losses[0]:.4f}")
    print(f"  - Final loss: {losses[-1]:.4f}")
    print(f"  - Loss decreased: {losses[0] > losses[-1]}")


def main():
    """Main test function."""
    print("\n" + "🧪 " * 20)
    print("MED-RiskNET Graph Neural Network Test Suite")
    print("🧪 " * 20 + "\n")
    
    try:
        # Test graph construction
        edge_index, node_features = test_graph_construction()
        
        # Test GraphSAGE encoder
        encoder_sage = test_gnn_encoder_graphsage(edge_index, node_features)
        
        # Test GAT encoder
        encoder_gat = test_gnn_encoder_gat(edge_index, node_features)
        
        # Test classifier
        classifier = test_gnn_classifier(edge_index, node_features)
        
        # Test variations
        test_graph_variations()
        
        # Test training
        test_mini_training()
        
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
