"""
Dataset classes for MED-RiskNet
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from torchvision import transforms
from PIL import Image
from typing import Optional, Dict, List, Tuple
from pathlib import Path

class TabularDataset(Dataset):
    """
    PyTorch Dataset for tabular clinical data.
    
    Handles:
    - Missing value imputation (median for numeric)
    - Standardization of numeric features
    - One-hot encoding of categorical features (e.g., sex)
    - Blood pressure parsing (systolic/diastolic split)
    
    Args:
        csv_path: Path to the CSV file
        numeric_features: List of numeric feature column names
        categorical_features: List of categorical feature column names
        target_column: Name of the target column
        scaler: Pre-fitted StandardScaler (optional, for test set)
        imputer: Pre-fitted SimpleImputer (optional, for test set)
        categories: Pre-fitted category mappings (optional, for test set)
        is_train: Whether this is training data (fit transformers)
    """

    def __init__(
        self,
        csv_path: str,
        numeric_features: List[str],
        categorical_features: List[str],
        target_column: str,
        scaler: Optional[StandardScaler] = None,
        imputer: Optional[SimpleImputer] = None,
        categories: Optional[Dict[str, np.ndarray]] = None,
        is_train: bool = True
    ):
        self.csv_path = Path(csv_path)
        self.numeric_features = numeric_features.copy()  # Copy to avoid modifying original
        self.categorical_features = categorical_features
        self.target_column = target_column
        self.is_train = is_train
        
        # Read CSV
        self.df = pd.read_csv(csv_path)
        
        # Store patient IDs if available (don't use as features)
        if 'patient_id' in self.df.columns:
            self.patient_ids = self.df['patient_id'].values
        else:
            self.patient_ids = None
        
        # Parse blood pressure into systolic/diastolic if present
        if 'bp' in self.df.columns:
            self.df[['bp_systolic', 'bp_diastolic']] = self.df['bp'].str.split('/', expand=True).astype(float)
            self.df.drop('bp', axis=1, inplace=True)
            # Update numeric features list
            if 'bp' in self.numeric_features:
                self.numeric_features.remove('bp')
                self.numeric_features.extend(['bp_systolic', 'bp_diastolic'])
        
        # Extract targets
        if target_column in self.df.columns:
            self.targets = self.df[target_column].values
            self.has_targets = True
        else:
            self.targets = None
            self.has_targets = False
        
        # Initialize or use provided transformers
        if is_train:
            self.imputer = SimpleImputer(strategy='median')
            self.scaler = StandardScaler()
            self.categories_ = {}  # Initialize empty dict for training
        else:
            assert scaler is not None and imputer is not None, \
                "Scaler and imputer must be provided for test/validation set"
            assert categories is not None, \
                "Categories must be provided for test/validation set"
            self.scaler = scaler
            self.imputer = imputer
            self.categories_ = categories
        
        # Preprocess features
        self.features = self._preprocess()
        self.feature_dim = self.features.shape[1]
    
    def _preprocess(self) -> np.ndarray:
        """
        Preprocess tabular data:
        1. Parse blood pressure into systolic/diastolic (already done in __init__)
        2. Impute missing values (median for numeric)
        3. Standardize numeric features
        4. One-hot encode categorical features
        
        Returns:
            Preprocessed feature array of shape (n_samples, n_features)
        """
        processed_features = []
        
        # Process numeric features
        if self.numeric_features:
            numeric_data = self.df[self.numeric_features].values
            
            # Impute missing values
            if self.is_train:
                numeric_data = self.imputer.fit_transform(numeric_data)
            else:
                numeric_data = self.imputer.transform(numeric_data)
            
            # Standardize
            if self.is_train:
                numeric_data = self.scaler.fit_transform(numeric_data)
            else:
                numeric_data = self.scaler.transform(numeric_data)
            
            processed_features.append(numeric_data)
        
        # Process categorical features (one-hot encoding)
        if self.categorical_features:
            for cat_feature in self.categorical_features:
                # Get or store unique categories
                if self.is_train:
                    # Store categories for this feature
                    self.categories_[cat_feature] = self.df[cat_feature].unique()
                
                # Get categories (from training set if test)
                unique_cats = self.categories_[cat_feature]
                
                # One-hot encode
                cat_data = self.df[cat_feature].values
                
                # Create one-hot matrix
                one_hot = np.zeros((len(cat_data), len(unique_cats)))
                for idx, cat in enumerate(unique_cats):
                    one_hot[cat_data == cat, idx] = 1
                
                processed_features.append(one_hot)
        
        # Concatenate all features
        if processed_features:
            features = np.concatenate(processed_features, axis=1)
        else:
            raise ValueError("No features to process")
        
        return features.astype(np.float32)
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
                - 'features': Preprocessed feature tensor
                - 'target': Target label (if available)
                - 'patient_id': Patient ID (if available)
        """
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        
        result = {'features': features}
        
        if self.has_targets:
            target = torch.tensor(self.targets[idx], dtype=torch.long)
            result['target'] = target
        
        if self.patient_ids is not None:
            result['patient_id'] = self.patient_ids[idx]
        
        return result
    
    def get_scaler(self) -> StandardScaler:
        """Return the fitted scaler (for use with test set).""" 
        return self.scaler
    
    def get_imputer(self) -> SimpleImputer:
        """Return the fitted imputer (for use with test set)."""
        return self.imputer
    
    def get_categories(self) -> Dict[str, np.ndarray]:
        """Return the category mappings (for use with test set)."""
        return self.categories_


class ImageDataset(Dataset):
    """
    PyTorch Dataset for medical images (e.g., chest X-rays).
    
    Handles:
    - Loading images by patient_id
    - Resizing to 224x224 (standard for ImageNet pretrained models)
    - Normalization using ImageNet statistics
    - Optional data augmentation
    
    Args:
        image_dir: Directory containing images
        patient_ids: List of patient IDs
        image_extension: File extension (default: '.png')
        transform: Custom transform pipeline (optional)
        target_size: Target image size (default: 224)
        augment: Whether to apply data augmentation (training only)
    """
    
    def __init__(
        self,
        image_dir: str,
        patient_ids: List[str],
        image_extension: str = '.png',
        transform: Optional[callable] = None,
        target_size: int = 224,
        augment: bool = False
    ):
        self.image_dir = Path(image_dir)
        self.patient_ids = patient_ids
        self.image_extension = image_extension
        self.target_size = target_size
        
        # ImageNet normalization statistics
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Build transform pipeline
        if transform is not None:
            self.transform = transform
        else:
            transform_list = []
            
            # Resize
            transform_list.append(transforms.Resize((target_size, target_size)))
            
            # Augmentation (training only)
            if augment:
                transform_list.extend([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=10),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2)
                ])
            
            # Convert to tensor and normalize
            transform_list.extend([
                transforms.ToTensor(),
                self.normalize
            ])
            
            self.transform = transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.patient_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single image from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
                - 'image': Transformed image tensor [3, 224, 224]
                - 'patient_id': Patient ID string
        """
        patient_id = self.patient_ids[idx]
        
        # Construct image path
        image_path = self.image_dir / f"{patient_id}_radiology{self.image_extension}"
        
        # Fallback to generic sample image if patient-specific doesn't exist
        if not image_path.exists():
            image_path = self.image_dir / f"sample_radiology_image{self.image_extension}"
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise FileNotFoundError(f"Could not load image for patient {patient_id}: {e}")
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        return {
            'image': image_tensor,
            'patient_id': patient_id
        }