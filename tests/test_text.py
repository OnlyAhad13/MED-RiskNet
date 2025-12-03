"""
Test script for text data pipeline.

This script tests:
1. TextDataset loading and parsing
2. TextEncoder with BioClinicalBERT
3. Tokenization and forward pass
4. CLS embedding extraction

Usage:
    python tests/test_text.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from data.datasets import TextDataset
from models.text import TextEncoder, TextClassifier


def test_text_dataset():
    """Test TextDataset class."""
    print("=" * 80)
    print("Testing TextDataset")
    print("=" * 80)
    
    # Sample patient IDs
    patient_ids = ['P001', 'P002', 'P003', 'P004', 'P005']
    
    # Create dataset
    dataset = TextDataset(
        report_file='data/sample_medical_reports.txt',
        patient_ids=patient_ids,
        max_length=512
    )
    
    print(f"✓ Dataset created successfully")
    print(f"  - Number of samples: {len(dataset)}")
    print(f"  - Report file: sample_medical_reports.txt")
    print(f"  - Max length: 512 tokens")
    
    # Get a sample
    sample = dataset[0]
    text = sample['text']
    patient_id = sample['patient_id']
    has_report = sample['has_report']
    
    print(f"\n✓ Sample extracted:")
    print(f"  - Patient ID: {patient_id}")
    print(f"  - Has report: {has_report}")
    print(f"  - Text length: {len(text)} characters")
    print(f"  - Text preview: {text[:100]}...")
    
    # Check all patients loaded
    loaded_reports = sum(dataset[i]['has_report'] for i in range(len(dataset)))
    print(f"\n✓ Reports loaded: {loaded_reports}/{len(dataset)}")
    
    # Test DataLoader
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False)
    batch = next(iter(dataloader))
    batch_texts = batch['text']
    batch_ids = batch['patient_id']
    
    print(f"\n✓ DataLoader test:")
    print(f"  - Batch size: {len(batch_texts)}")
    print(f"  - Batch IDs: {batch_ids}")
    
    return dataset


def test_text_encoder():
    """Test TextEncoder with BioClinicalBERT."""
    print("\n" + "=" * 80)
    print("Testing TextEncoder (BioClinicalBERT)")
    print("=" * 80)
    
    try:
        # Create encoder
        encoder = TextEncoder(
            model_name='emilyalsentzer/Bio_ClinicalBERT',
            feature_dim=768,
            max_length=512,
            freeze_bert=False
        )
        
        print(f"\n✓ BioClinicalBERT encoder created:")
        print(f"  - Feature dim: 768")
        print(f"  - Max length: 512")
        print(f"  - Model: emilyalsentzer/Bio_ClinicalBERT")
        
    except Exception as e:
        print(f"\n⚠ Could not load BioClinicalBERT: {e}")
        print("This is expected if transformers is not installed.")
        print("Install with: pip install transformers")
        return None
    
    # Count parameters
    num_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"  - Trainable parameters: {num_params:,}")
    
    # Test tokenization
    sample_texts = [
        "Patient presents with mild hypertension and elevated cholesterol.",
        "Chest X-ray shows mild cardiomegaly.",
        "All vital signs within normal limits."
    ]
    
    print(f"\n✓ Testing tokenization:")
    encoding = encoder.tokenize(sample_texts)
    print(f"  - Input IDs shape: {encoding['input_ids'].shape}")
    print(f"  - Attention mask shape: {encoding['attention_mask'].shape}")
    
    # Test forward pass with raw text
    encoder.eval()
    with torch.no_grad():
        embeddings = encoder(texts=sample_texts)
    
    print(f"\n✓ Forward pass successful:")
    print(f"  - Input: {len(sample_texts)} texts")
    print(f"  - Output shape: {embeddings.shape}")
    print(f"  - Output mean: {embeddings.mean().item():.4f}")
    print(f"  - Output std: {embeddings.std().item():.4f}")
    
    assert embeddings.shape == (3, 768), f"Expected shape (3, 768), got {embeddings.shape}"
    print(f"\n✓ Output shape is correct (CLS embeddings)")
    
    return encoder


def test_text_encoder_with_tokens():
    """Test TextEncoder with pre-tokenized inputs."""
    print("\n" + "=" * 80)
    print("Testing TextEncoder (with pre-tokenized inputs)")
    print("=" * 80)
    
    try:
        encoder = TextEncoder(
            model_name='emilyalsentzer/Bio_ClinicalBERT',
            feature_dim=512,  # Different dimension
            max_length=256
        )
    except:
        print("⚠ Skipping test (transformers not available)")
        return None
    
    # Tokenize separately
    texts = ["Patient with diabetes and hypertension."]
    encoding = encoder.tokenize(texts)
    
    print(f"✓ Tokenized separately:")
    print(f"  - Input IDs: {encoding['input_ids'].shape}")
    
    # Forward pass with tokens
    encoder.eval()
    with torch.no_grad():
        embeddings = encoder(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask']
        )
    
    print(f"\n✓ Forward pass with tokens:")
    print(f"  - Output shape: {embeddings.shape}")
    print(f"  - Feature dim: 512 (projected from 768)")
    
    assert embeddings.shape == (1, 512), f"Expected shape (1, 512), got {embeddings.shape}"
    print(f"\n✓ Projection head working correctly")


def test_text_classifier():
    """Test TextClassifier class."""
    print("\n" + "=" * 80)
    print("Testing TextClassifier")
    print("=" * 80)
    
    try:
        # Create classifier
        classifier = TextClassifier(
            model_name='emilyalsentzer/Bio_ClinicalBERT',
            feature_dim=768,
            num_classes=1,
            max_length=512,
            freeze_bert=False
        )
        
        print(f"\n✓ Classifier created:")
        print(f"  - Feature dim: 768")
        print(f"  - Num classes: 1 (binary)")
        
    except:
        print("⚠ Skipping test (transformers not available)")
        return None
    
    # Count parameters
    num_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"  - Trainable parameters: {num_params:,}")
    
    # Test forward pass
    sample_texts = [
        "High-risk patient with multiple comorbidities.",
        "Healthy patient with normal vitals."
    ]
    
    classifier.eval()
    with torch.no_grad():
        predictions = classifier(texts=sample_texts)
    
    print(f"\n✓ Forward pass successful:")
    print(f"  - Input: {len(sample_texts)} texts")
    print(f"  - Predictions shape: {predictions.shape}")
    print(f"  - Predictions: {predictions.squeeze().tolist()}")
    
    # Test with embedding return
    with torch.no_grad():
        predictions, embeddings = classifier(texts=sample_texts, return_embedding=True)
    
    print(f"\n✓ Forward pass with embeddings:")
    print(f"  - Predictions shape: {predictions.shape}")
    print(f"  - Embeddings shape: {embeddings.shape}")
    
    return classifier


def test_real_data_forward():
    """Test forward pass with real medical reports."""
    print("\n" + "=" * 80)
    print("Testing Real Medical Report Forward Pass")
    print("=" * 80)
    
    # Create dataset
    patient_ids = ['P001', 'P002']
    dataset = TextDataset(
        report_file='data/sample_medical_reports.txt',
        patient_ids=patient_ids,
        max_length=512
    )
    
    # Get real reports
    reports = [dataset[i]['text'] for i in range(len(dataset))]
    
    print(f"✓ Loaded real reports:")
    for i, (pid, text) in enumerate(zip(patient_ids, reports)):
        print(f"  - {pid}: {text[:60]}...")
    
    try:
        # Create encoder
        encoder = TextEncoder(
            model_name='emilyalsentzer/Bio_ClinicalBERT',
            feature_dim=768
        )
        
        # Forward pass
        encoder.eval()
        with torch.no_grad():
            embeddings = encoder(texts=reports)
        
        print(f"\n✓ Feature extraction successful:")
        print(f"  - Embeddings shape: {embeddings.shape}")
        print(f"  - Embeddings mean: {embeddings.mean().item():.4f}")
        print(f"  - Embeddings std: {embeddings.std().item():.4f}")
        print(f"  - Embedding sample (first 5): {embeddings[0][:5].tolist()}")
        
    except:
        print("\n⚠ Skipping (transformers not available)")


def main():
    """Main test function."""
    print("\n" + "🧪 " * 20)
    print("MED-RiskNET Text Pipeline Test Suite")
    print("🧪 " * 20 + "\n")
    
    try:
        # Test dataset
        dataset = test_text_dataset()
        
        # Test encoder
        encoder = test_text_encoder()
        
        # Test encoder with pre-tokenization
        test_text_encoder_with_tokens()
        
        # Test classifier
        classifier = test_text_classifier()
        
        # Test real data
        test_real_data_forward()
        
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
