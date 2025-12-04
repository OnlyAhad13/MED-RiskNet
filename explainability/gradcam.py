"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for image explainability.

Generates heatmaps showing which regions of an image are important for predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Tuple
import matplotlib.pyplot as plt


class GradCAM:
    """
    Grad-CAM implementation for CNN explainability.
    
    Generates class activation maps showing which image regions
    contribute most to predictions.
    
    Args:
        model: CNN model (must have feature extraction layers)
        target_layer: Layer to extract gradients from (e.g., last conv layer)
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Hook to save forward activations."""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(
        self,
        input_image: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_image: Input image tensor [1, 3, H, W]
            target_class: Target class index (None for highest prediction)
        
        Returns:
            CAM heatmap [H, W]
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # [C, H', W']
        activations = self.activations[0]  # [C, H', W']
        
        # Global average pooling of gradients (weights)
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]
        
        # Weighted combination of activations
        cam = (weights * activations).sum(dim=0)  # [H', W']
        
        # ReLU to keep only positive influences
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()
    
    def visualize_cam(
        self,
        input_image: torch.Tensor,
        cam: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay CAM heatmap on original image.
        
        Args:
            input_image: Original image tensor [1, 3, H, W]
            cam: CAM heatmap [H', W']
            alpha: Overlay transparency
            colormap: OpenCV colormap
        
        Returns:
            Overlaid image [H, W, 3]
        """
        # Get image dimensions
        h, w = input_image.shape[2:]
        
        # Resize CAM to match image size
        cam_resized = cv2.resize(cam, (w, h))
        
        # Convert to RGB heatmap
        cam_colored = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8),
            colormap
        )
        cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
        
        # Convert input image to numpy
        img = input_image[0].permute(1, 2, 0).cpu().numpy()
        
        # Denormalize (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img * std + mean).clip(0, 1)
        img = (img * 255).astype(np.uint8)
        
        # Overlay
        overlaid = cv2.addWeighted(img, 1 - alpha, cam_colored, alpha, 0)
        
        return overlaid
    
    def save_visualization(
        self,
        input_image: torch.Tensor,
        cam: np.ndarray,
        save_path: str,
        title: str = "Grad-CAM Visualization"
    ):
        """
        Save CAM visualization to file.
        
        Args:
            input_image: Input image tensor
            cam: CAM heatmap
            save_path: Output file path
            title: Plot title
        """
        overlaid = self.visualize_cam(input_image, cam)
        
        plt.figure(figsize=(10, 5))
        
        # Original image
        plt.subplot(1, 2, 1)
        img = input_image[0].permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img * std + mean).clip(0, 1)
        plt.imshow(img)
        plt.title("Original Image")
        plt.axis('off')
        
        # Grad-CAM overlay
        plt.subplot(1, 2, 2)
        plt.imshow(overlaid)
        plt.title(title)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Grad-CAM saved to {save_path}")


def get_gradcam_for_medrisknet(
    model,
    image: torch.Tensor,
    tabular: Optional[torch.Tensor] = None
) -> Tuple[np.ndarray, float]:
    """
    Get Grad-CAM for MED-RiskNET image encoder.
    
    Args:
        model: MedRiskNet model
        image: Image tensor [1, 3, 224, 224]
        tabular: Optional tabular features
    
    Returns:
        CAM heatmap, prediction score
    """
    # Get last conv layer of image encoder
    # For ResNet: layer4[-1].conv3
    if hasattr(model.image_encoder.backbone, 'layer4'):
        target_layer = model.image_encoder.backbone.layer4[-1]
    else:
        raise ValueError("Unsupported CNN architecture")
    
    # Create Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # Generate CAM
    cam = gradcam.generate_cam(image)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(tabular=tabular, image=image)
        prediction = outputs['classification'].item()
    
    return cam, prediction


__all__ = ['GradCAM', 'get_gradcam_for_medrisknet']
