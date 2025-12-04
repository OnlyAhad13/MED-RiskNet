"""
SHAP (SHapley Additive exPlanations) for tabular data explainability.

Computes feature importance using Shapley values.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not installed. Install with: pip install shap")


class SHAPExplainer:
    """
    SHAP explainer wrapper for tabular data.
    
    Uses KernelSHAP to explain predictions based on tabular features.
    
    Args:
        model: Trained model
        background_data: Background dataset for SHAP [n_samples, n_features]
        feature_names: List of feature names
    """
    
    def __init__(
        self,
        model,
        background_data: np.ndarray,
        feature_names: Optional[List[str]] = None
    ):
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required. Install with: pip install shap")
        
        self.model = model
        self.feature_names = feature_names or [f"Feature {i}" for i in range(background_data.shape[1])]
        
        # Create prediction function for SHAP
        def predict_fn(X):
            """Prediction function for SHAP."""
            X_tensor = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                outputs = model(tabular=X_tensor)
                probs = outputs['classification'].cpu().numpy()
            # Ensure output is always 1D (SHAP requirement)
            return np.atleast_1d(probs.squeeze())
        
        self.predict_fn = predict_fn
        
        # Initialize SHAP explainer
        self.explainer = shap.KernelExplainer(
            self.predict_fn,
            background_data
        )
    
    def explain(
        self,
        X: np.ndarray,
        nsamples: int = 100
    ) -> shap.Explanation:
        """
        Compute SHAP values for input samples.
        
        Args:
            X: Input features [n_samples, n_features]
            nsamples: Number of samples for KernelSHAP
        
        Returns:
            SHAP explanation object
        """
        shap_values = self.explainer.shap_values(X, nsamples=nsamples)
        
        return shap.Explanation(
            values=shap_values,
            base_values=self.explainer.expected_value,
            data=X,
            feature_names=self.feature_names
        )
    
    def plot_waterfall(
        self,
        explanation: shap.Explanation,
        sample_idx: int = 0,
        save_path: Optional[str] = None
    ):
        """
        Plot waterfall plot for a single prediction.
        
        Shows how each feature contributed to the prediction.
        
        Args:
            explanation: SHAP explanation
            sample_idx: Sample index to explain
            save_path: Optional path to save figure
        """
        shap.waterfall_plot(
            shap.Explanation(
                values=explanation.values[sample_idx],
                base_values=explanation.base_values,
                data=explanation.data[sample_idx],
                feature_names=self.feature_names
            ),
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Waterfall plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_summary(
        self,
        explanation: shap.Explanation,
        save_path: Optional[str] = None
    ):
        """
        Plot summary of feature importance across all samples.
        
        Args:
            explanation: SHAP explanation
            save_path: Optional path to save figure
        """
        shap.summary_plot(
            explanation.values,
            explanation.data,
            feature_names=self.feature_names,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Summary plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_force(
        self,
        explanation: shap.Explanation,
        sample_idx: int = 0,
        save_path: Optional[str] = None
    ):
        """
        Plot force plot for a single prediction.
        
        Args:
            explanation: SHAP explanation
            sample_idx: Sample index
            save_path: Optional path to save HTML
        """
        force_plot = shap.force_plot(
            explanation.base_values,
            explanation.values[sample_idx],
            explanation.data[sample_idx],
            feature_names=self.feature_names
        )
        
        if save_path:
            shap.save_html(save_path, force_plot)
            print(f"✓ Force plot saved to {save_path}")
        else:
            return force_plot
    
    def get_feature_importance(
        self,
        explanation: shap.Explanation
    ) -> pd.DataFrame:
        """
        Get feature importance ranking.
        
        Args:
            explanation: SHAP explanation
        
        Returns:
            DataFrame with features and importance scores
        """
        importance = np.abs(explanation.values).mean(axis=0)
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        return df


def explain_medrisknet_tabular(
    model,
    tabular_data: np.ndarray,
    feature_names: Optional[List[str]] = None,
    background_size: int = 50
) -> SHAPExplainer:
    """
    Create SHAP explainer for MED-RiskNET tabular encoder.
    
    Args:
        model: MedRiskNet model
        tabular_data: Full tabular dataset [n_samples, n_features]
        feature_names: Feature names
        background_size: Number of background samples
    
    Returns:
        Fitted SHAP explainer
    """
    # Sample background data
    indices = np.random.choice(len(tabular_data), size=background_size, replace=False)
    background_data = tabular_data[indices]
    
    # Create explainer
    explainer = SHAPExplainer(model, background_data, feature_names)
    
    return explainer


__all__ = ['SHAPExplainer', 'explain_medrisknet_tabular']
