"""
Example usage of MED-RiskNET explainability tools.

Demonstrates:
- Grad-CAM for image explanations
- SHAP for tabular feature importance
- GNNExplainer for graph explanations

Usage:
    python explainability/examples.py
"""

import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from models.medrisknet import MedRiskNet
from data.multimodal_dataset import MultimodalDataset
from explainability.gradcam import get_gradcam_for_medrisknet, GradCAM
from explainability.shap_explainer import explain_medrisknet_tabular
from explainability.gnn_explainer import GNNExplainerWrapper
from data.graph_builder import PatientGraphBuilder


def example_gradcam():
    """Example: Grad-CAM for image explanations."""
    print("\n" + "=" * 80)
    print("Example 1: Grad-CAM for Image Explainability")
    print("=" * 80)
    
    try:
        # Load model
        print("Loading model...")
        model = MedRiskNet(
            tabular_input_dim=8,
            num_classes=1,
            fusion_hidden_dim=128,
            use_graph=False
        )
        model.eval()
        
        # Load sample image
        print("Loading sample image...")
        img_path = "data/sample_radiology_image.png"
        img = Image.open(img_path).convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(img).unsqueeze(0)
        
        # Dummy tabular for full forward pass
        tabular = torch.randn(1, 8)
        
        # Get last conv layer from ResNet backbone
        backbone = model.image_encoder.backbone
        if hasattr(backbone, 'layer4'):
            target_layer = backbone.layer4[-1]
        elif hasattr(model.image_encoder, 'features'):
            # For other architectures like EfficientNet
            target_layer = model.image_encoder.features[-1]
        else:
            print("Warning: CNN architecture not supported for Grad-CAM")
            print(f"  Available attributes: {dir(backbone)}")
            return
        
        # Create Grad-CAM  
        print("Generating Grad-CAM...")
        gradcam = GradCAM(model.image_encoder, target_layer)
        
        # Generate CAM
        cam = gradcam.generate_cam(image_tensor, target_class=0)
        
        print("✓ Grad-CAM generated successfully")
        print(f"  CAM shape: {cam.shape}")
        print(f"  CAM range: [{cam.min():.3f}, {cam.max():.3f}]")
        
        # Save visualization
        output_path = "gradcam_example.png"
        gradcam.save_visualization(
            image_tensor,
            cam,
            output_path,
            title="Grad-CAM: Important Regions"
        )
        
    except Exception as e:
        print(f"Grad-CAM example failed: {e}")
        import traceback
        traceback.print_exc()


def example_shap():
    """Example: SHAP for tabular feature importance."""
    print("\n" + "=" * 80)
    print("Example 2: SHAP for Tabular Feature Importance")
    print("=" * 80)
    
    try:
        # Load model
        print("Loading model...")
        model = MedRiskNet(
            tabular_input_dim=8,
            num_classes=1,
            fusion_hidden_dim=128,
            use_graph=False
        )
        model.eval()
        
        # Load dataset
        print("Loading dataset...")
        dataset = MultimodalDataset(
            csv_path='data/sample_patient_data.csv',
            image_dir='data',
            text_file='data/sample_medical_reports.txt'
        )
        
        # Get tabular data
        tabular_data = dataset.tabular_data
        feature_names = dataset.feature_cols
        
        print(f"✓ Loaded {len(dataset)} samples")
        print(f"  Features: {feature_names}")
        
        # Try to create SHAP explainer
        print("\nCreating SHAP explainer...")
        try:
            explainer = explain_medrisknet_tabular(
                model,
                tabular_data,
                feature_names,
                background_size=5
            )
            
            # Explain first 3 samples
            print("Computing SHAP values...")
            explanation = explainer.explain(tabular_data[:3], nsamples=50)
            
            print("✓ SHAP values computed successfully")
            
            # Get feature importance
            importance_df = explainer.get_feature_importance(explanation)
            
            print("\nFeature Importance Ranking:")
            print(importance_df.to_string(index=False))
            
            # Save plots
            print("\nSaving visualizations...")
            explainer.plot_waterfall(explanation, 0, "shap_waterfall.png")
            explainer.plot_summary(explanation, "shap_summary.png")
            
        except ImportError:
            print("SHAP not installed. Showing simulated results...")
            print("Install with: pip install shap")
            
            # Simulated importance
            importance = np.abs(np.random.randn(len(feature_names)))
            importance = importance / importance.sum()
            
            print(f"\nSimulated feature importance:")
            for feat, imp in sorted(zip(feature_names, importance), key=lambda x: -x[1]):
                print(f"  {feat}: {imp:.3f}")
                
    except Exception as e:
        print(f"SHAP example failed: {e}")
        import traceback
        traceback.print_exc()


def example_gnn_explainer():
    """Example: GNNExplainer for graph explanations."""
    print("\n" + "=" * 80)
    print("Example 3: GNNExplainer for Graph Explainability")
    print("=" * 80)
    
    try:
        # Load model
        print("Loading model...")
        model = MedRiskNet(
            tabular_input_dim=8,
            num_classes=1,
            fusion_hidden_dim=128,
            use_graph=True
        )
        model.eval()
        
        # Load dataset
        print("Loading dataset...")
        dataset = MultimodalDataset(
            csv_path='data/sample_patient_data.csv',
            image_dir='data',
            text_file='data/sample_medical_reports.txt'
        )
        
        # Build graph
        print("Building patient similarity graph...")
        tabular_data = torch.tensor(dataset.tabular_data, dtype=torch.float32)
        builder = PatientGraphBuilder(k_neighbors=3)
        edge_index = builder.build_graph_from_features(tabular_data)
        
        print(f"✓ Graph built:")
        print(f"  - Nodes: {len(dataset)} patients")
        print(f"  - Edges: {edge_index.shape[1]}")
        
        # Get patient to explain
        target_patient = 0
        neighbors = edge_index[1, edge_index[0] == target_patient].tolist()
        
        print(f"\nPatient {target_patient} connections:")
        print(f"  Connected to patients: {neighbors}")
        
        # Try GNNExplainer
        try:
            print("\nCreating GNNExplainer...")
            explainer = GNNExplainerWrapper(model.graph_encoder)
            
            print("Computing explanation...")
            explanation = explainer.explain_node(
                node_idx=target_patient,
                x=tabular_data,
                edge_index=edge_index
            )
            
            print("✓ GNN explanation generated")
            
            # Get important neighbors
            important_neighbors = explainer.get_important_neighbors(
                explanation,
                target_patient,
                edge_index,
                top_k=3
            )
            
            print("\nMost important neighbors:")
            for neighbor, importance in important_neighbors:
                print(f"  Patient {neighbor}: importance = {importance:.3f}")
            
            # Save visualization
            print("\nSaving subgraph visualization...")
            explainer.visualize_explanation(
                explanation,
                target_patient,
                edge_index,
                save_path="gnn_explanation.png",
                title=f"GNN Explanation for Patient {target_patient}"
            )
            
        except ImportError:
            print("PyTorch Geometric not installed.")
            print("Install with: pip install torch-geometric")
            print("\nShowing basic neighbor analysis...")
            
            # Simple neighbor importance (based on # of connections)
            neighbor_counts = {}
            for i in range(len(dataset)):
                neighbor_counts[i] = ((edge_index[0] == i) | (edge_index[1] == i)).sum().item()
            
            print(f"\nPatient connectivity (proxy for importance):")
            for neighbor in neighbors:
                print(f"  Patient {neighbor}: {neighbor_counts[neighbor]} connections")
                
    except Exception as e:
        print(f"GNN example failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all examples."""
    print("\n" + "🔍 " * 20)
    print("MED-RiskNET Explainability Examples")
    print("🔍 " * 20)
    
    try:
        # Grad-CAM example
        example_gradcam()
        
        # SHAP example
        example_shap()
        
        # GNNExplainer example
        example_gnn_explainer()
        
        print("\n" + "=" * 80)
        print("ALL EXAMPLES COMPLETED")
        print("=" * 80)
        
        print("\nSummary:")
        print("  ✓ Grad-CAM: Visualizes important image regions")
        print("  ✓ SHAP: Computes tabular feature importance")
        print("  ✓ GNNExplainer: Identifies important graph neighbors")
        
        print("\nNext steps:")
        print("  1. Install dependencies: pip install shap torch-geometric")
        print("  2. Train a model: python tests/test_train_smoke.py")
        print("  3. Run explainability on trained model")
        
    except Exception as e:
        print(f"\nExample failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
