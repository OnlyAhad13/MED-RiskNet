"""
GNNExplainer for graph neural network explainability.

Identifies important subgraphs and features for GNN predictions.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Optional, Tuple
import warnings

try:
    from torch_geometric.explain import Explainer, GNNExplainer as PyGGNNExplainer
    from torch_geometric.data import Data
    PYGED_AVAILABLE = True
except ImportError:
    PYGED_AVAILABLE = False
    warnings.warn("PyTorch Geometric not installed. GNNExplainer unavailable.")


class GNNExplainerWrapper:
    """
    GNNExplainer wrapper for graph explainability.
    
    Identifies which nodes and edges are most important for predictions.
    
    Args:
        model: GNN model
        num_hops: Number of hops for subgraph
        epochs: Training epochs for explainer
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_hops: int = 2,
        epochs: int = 100
    ):
        if not PYGED_AVAILABLE:
            raise ImportError("PyTorch Geometric required for GNNExplainer")
        
        self.model = model
        self.num_hops = num_hops
        
        # Create explainer
        self.explainer = Explainer(
            model=model,
            algorithm=PyGGNNExplainer(epochs=epochs),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='probs'
            )
        )
    
    def explain_node(
        self,
        node_idx: int,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ):
        """
        Explain prediction for a specific node.
        
        Args:
            node_idx: Node to explain
            x: Node features [num_nodes, num_features]
            edge_index: Edge indices [2, num_edges]
        
        Returns:
            Explanation object with masks
        """
        explanation = self.explainer(
            x=x,
            edge_index=edge_index,
            index=node_idx
        )
        
        return explanation
    
    def visualize_explanation(
        self,
        explanation,
        node_idx: int,
        edge_index: torch.Tensor,
        save_path: Optional[str] = None,
        title: str = "GNN Explanation"
    ):
        """
        Visualize explanation as a graph.
        
        Args:
            explanation: Explainer output
            node_idx: Target node
            edge_index: Edge indices
            save_path: Optional save path
            title: Plot title
        """
        # Get masks
        node_mask = explanation.node_mask if hasattr(explanation, 'node_mask') else None
        edge_mask = explanation.edge_mask if hasattr(explanation, 'edge_mask') else None
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add edges
        edges = edge_index.t().cpu().numpy()
        for i, (src, dst) in enumerate(edges):
            weight = edge_mask[i].item() if edge_mask is not None else 1.0
            G.add_edge(src, dst, weight=weight)
        
        # Layout
        pos = nx.spring_layout(G, seed=42)
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        # Draw edges with weights
        if edge_mask is not None:
            edge_colors = [G[u][v]['weight'] for u, v in G.edges()]
            nx.draw_networkx_edges(
                G, pos,
                edge_color=edge_colors,
                edge_cmap=plt.cm.Reds,
                width=2,
                edge_vmin=0,
                edge_vmax=1
            )
        else:
            nx.draw_networkx_edges(G, pos, width=1, alpha=0.5)
        
        # Draw nodes
        node_colors = ['red' if n == node_idx else 'lightblue' for n in G.nodes()]
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=500,
            alpha=0.9
        )
        
        # Labels
        nx.draw_networkx_labels(G, pos, font_size=10)
        
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ GNN explanation saved to {save_path}")
        else:
            plt.show()
    
    def get_important_neighbors(
        self,
        explanation,
        node_idx: int,
        edge_index: torch.Tensor,
        top_k: int = 5
    ) -> list:
        """
        Get most important neighbor nodes.
        
        Args:
            explanation: Explainer output
            node_idx: Target node
            edge_index: Edge indices
            top_k: Number of neighbors to return
        
        Returns:
            List of (neighbor_idx, importance_score) tuples
        """
        edge_mask = explanation.edge_mask
        
        # Find edges connected to target node
        edges = edge_index.t()
        connected_edges = (edges[:, 0] == node_idx) | (edges[:, 1] == node_idx)
        
        # Get neighbors and their importance
        neighbors = []
        for i, is_connected in enumerate(connected_edges):
            if is_connected:
                src, dst = edges[i]
                neighbor = dst if src == node_idx else src
                importance = edge_mask[i].item()
                neighbors.append((neighbor.item(), importance))
        
        # Sort by importance
        neighbors.sort(key=lambda x: x[1], reverse=True)
        
        return neighbors[:top_k]


def explain_medrisknet_graph(
    model,
    graph_data: Data,
    node_idx: int = 0
):
    """
    Explain MED-RiskNET graph prediction for a patient.
    
    Args:
        model: MedRiskNet model
        graph_data: Graph data (x, edge_index)
        node_idx: Patient node to explain
    
    Returns:
        Explanation object
    """
    # Extract GNN encoder
    gnn_model = model.graph_encoder
    
    # Create explainer
    explainer_wrapper = GNNExplainerWrapper(gnn_model)
    
    # Explain
    explanation = explainer_wrapper.explain_node(
        node_idx=node_idx,
        x=graph_data.x,
        edge_index=graph_data.edge_index
    )
    
    return explanation, explainer_wrapper


__all__ = ['GNNExplainerWrapper', 'explain_medrisknet_graph']
