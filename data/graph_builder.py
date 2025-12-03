"""
Graph construction utilities for MED-RiskNET.

This module provides utilities to build patient similarity graphs
from tabular clinical data using k-nearest neighbors.
"""

import pandas as pd
import numpy as np
import torch
from typing import Tuple, Optional
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


class PatientGraphBuilder:
    """
    Build patient similarity graphs from tabular data.
    
    Constructs a k-nearest neighbor graph where:
    - Nodes: Individual patients
    - Edges: Similarity connections (cosine similarity > threshold)
    
    Args:
        k_neighbors: Number of nearest neighbors per patient
        similarity_threshold: Minimum similarity for edge creation
        include_self_loops: Whether to add self-loops
    """
    
    def __init__(
        self,
        k_neighbors: int = 3,
        similarity_threshold: float = 0.0,
        include_self_loops: bool = False
    ):
        self.k_neighbors = k_neighbors
        self.similarity_threshold = similarity_threshold
        self.include_self_loops = include_self_loops
    
    def build_graph_from_csv(
        self,
        csv_path: str,
        feature_columns: Optional[list] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build graph from CSV file.
        
        Args:
            csv_path: Path to CSV file
            feature_columns: List of column names to use as features
                           (if None, uses all numeric columns except patient_id, label, and categorical columns)
        
        Returns:
            edge_index: Edge indices [2, num_edges]
            node_features: Node feature matrix [num_nodes, num_features]
        """
        # Load data
        df = pd.read_csv(csv_path)
        
        # Extract features
        if feature_columns is None:
            # Exclude non-feature columns
            exclude_cols = ['patient_id', 'label', 'sex']  # sex is categorical
            feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        # Build feature matrix, handling special cases
        features_list = []
        
        for col in feature_columns:
            if col == 'bp':
                # Parse blood pressure (format: "130/85" -> [130, 85])
                bp_values = df[col].str.split('/', expand=True).astype(float).values
                features_list.append(bp_values)
            else:
                # Regular numeric column
                features_list.append(df[[col]].values)
        
        # Combine all features
        features = np.hstack(features_list)
        
        # Build graph
        edge_index = self.build_knn_graph(features)
        node_features = torch.tensor(features, dtype=torch.float32)
        
        return edge_index, node_features
    
    def build_knn_graph(self, features: np.ndarray) -> torch.Tensor:
        """
        Build k-nearest neighbor graph using cosine similarity.
        
        Args:
            features: Feature matrix [num_nodes, num_features]
        
        Returns:
            edge_index: Edge indices [2, num_edges]
        """
        # Compute pairwise cosine similarity
        similarity_matrix = cosine_similarity(features)
        
        # For each patient, find k nearest neighbors
        num_patients = features.shape[0]
        edges = []
        
        for i in range(num_patients):
            # Get similarities for patient i
            similarities = similarity_matrix[i]
            
            # Set self-similarity to -inf to exclude it
            similarities[i] = -np.inf
            
            # Get indices of k most similar patients
            k_nearest = np.argsort(similarities)[-self.k_neighbors:]
            
            # Filter by threshold
            for j in k_nearest:
                if similarity_matrix[i, j] >= self.similarity_threshold:
                    edges.append([i, j])
        
        # Add self-loops if requested
        if self.include_self_loops:
            for i in range(num_patients):
                edges.append([i, i])
        
        # Convert to edge_index format [2, num_edges]
        if len(edges) > 0:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
        else:
            # Empty graph
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        return edge_index
    
    def build_graph_from_features(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Build k-NN graph from feature tensor.
        
        Args:
            features: Feature tensor [num_nodes, num_features]
        
        Returns:
            edge_index: Edge indices [2, num_edges]
        """
        features_np = features.cpu().numpy() if isinstance(features, torch.Tensor) else features
        return self.build_knn_graph(features_np)
    
    def get_graph_statistics(self, edge_index: torch.Tensor, num_nodes: int) -> dict:
        """
        Compute graph statistics.
        
        Args:
            edge_index: Edge indices [2, num_edges]
            num_nodes: Number of nodes
        
        Returns:
            Dictionary with graph statistics
        """
        num_edges = edge_index.shape[1]
        
        # Compute degree distribution
        degrees = torch.zeros(num_nodes, dtype=torch.long)
        for i in range(num_edges):
            degrees[edge_index[0, i]] += 1
        
        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'avg_degree': degrees.float().mean().item(),
            'max_degree': degrees.max().item(),
            'min_degree': degrees.min().item(),
            'density': num_edges / (num_nodes * (num_nodes - 1))
        }
