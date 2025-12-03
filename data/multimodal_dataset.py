"""
Multimodal dataset for MED-RiskNET training.

Combines tabular, image, and text data.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Optional
from PIL import Image
import torchvision.transforms as transforms


class MultimodalDataset(Dataset):
    """
    Combined dataset for all modalities.
    
    Args:
        csv_path: Path to CSV with patient data
        image_dir: Directory with images
        text_file: Path to medical reports
        patient_ids: List of patient IDs to load
        image_transform: Image transformations
    """
    
    def __init__(
        self,
        csv_path: str,
        image_dir: Optional[str] = None,
        text_file: Optional[str] = None,
        patient_ids: Optional[List[str]] = None,
        image_transform: Optional[transforms.Compose] = None
    ):
        # Load tabular data
        self.df = pd.read_csv(csv_path)
        
        if patient_ids is not None:
            self.df = self.df[self.df['patient_id'].isin(patient_ids)]
        
        self.patient_ids = self.df['patient_id'].tolist()
        
        # Image setup
        self.image_dir = Path(image_dir) if image_dir else None
        if image_transform is None:
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.image_transform = image_transform
        
        # Text setup
        self.text_file = Path(text_file) if text_file else None
        self.texts = {}
        if self.text_file and self.text_file.exists():
            self._load_texts()
        
        # Prepare tabular features
        self._prepare_tabular_features()
    
    def _load_texts(self):
        """Load medical reports."""
        with open(self.text_file, 'r') as f:
            content = f.read()
        
        for entry in content.strip().split('\n\n'):
            if ':' in entry:
                parts = entry.split(':', 1)
                if len(parts) == 2:
                    pid = parts[0].strip()
                    text = parts[1].strip()
                    self.texts[pid] = text
    
    def _prepare_tabular_features(self):
        """Prepare tabular features."""
        # Exclude non-feature columns (sex and bp handled separately)
        feature_cols = [col for col in self.df.columns 
                       if col not in ['patient_id', 'label', 'sex', 'bp']]
        
        # Handle blood pressure (split "130/85" -> [130, 85])
        if 'bp' in self.df.columns:
            bp_split = self.df['bp'].str.split('/', expand=True).astype(float)
            self.df['bp_systolic'] = bp_split[0]
            self.df['bp_diastolic'] = bp_split[1]
            feature_cols.extend(['bp_systolic', 'bp_diastolic'])
        
        # Handle sex (one-hot encode: M -> [1, 0], F -> [0, 1])
        if 'sex' in self.df.columns:
            sex_encoded = pd.get_dummies(self.df['sex'], prefix='sex')
            self.df[sex_encoded.columns] = sex_encoded.astype('float32')
            feature_cols.extend(sex_encoded.columns.tolist())
        
        self.feature_cols = feature_cols
        self.tabular_data = self.df[feature_cols].values.astype('float32')
        self.labels = self.df['label'].values if 'label' in self.df.columns else None
    
    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        
        # Tabular features
        tabular = torch.tensor(self.tabular_data[idx], dtype=torch.float32)
        
        # Label
        label = self.labels[idx] if self.labels is not None else 0
        
        # Image (if available)
        image = None
        if self.image_dir:
            image_path = self.image_dir / f"{patient_id}_radiology.png"
            if not image_path.exists():
                image_path = self.image_dir / "sample_radiology_image.png"
            
            try:
                img = Image.open(image_path).convert('RGB')
                image = self.image_transform(img)
            except:
                # Return zeros if image loading fails
                image = torch.zeros(3, 224, 224)
        
        # Text (if available)
        text = self.texts.get(patient_id, "")
        
        return {
            'patient_id': patient_id,
            'tabular': tabular,
            'image': image,
            'text': text,
            'label': label
        }


def collate_multimodal(batch):
    """
    Custom collate function for multimodal batches.
    
    Handles: - Stacking tensors
    - Collecting text strings
    - Handling missing modalities
    """
    patient_ids = [item['patient_id'] for item in batch]
    tabular = torch.stack([item['tabular'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32)
    
    # Images (may be None)
    images = [item['image'] for item in batch]
    if any(img is not None for img in images):
        images = torch.stack([img if img is not None else torch.zeros(3, 224, 224) 
                             for img in images])
    else:
        images = None
    
    # Texts (list of strings)
    texts = [item['text'] for item in batch]
    if not any(texts):
        texts = None
    
    return {
        'patient_id': patient_ids,
        'tabular': tabular,
        'image': images,
        'text': texts,
        'label': labels
    }
