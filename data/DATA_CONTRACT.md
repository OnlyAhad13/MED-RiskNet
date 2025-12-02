# MED-RiskNET Data Contract

## Overview
This document describes the data formats, schemas, expected datatypes, missing-value handling strategies, and graph construction rules for the MED-RiskNET project. This contract ensures consistency across data ingestion, processing, and model training pipelines.

---

## 1. Tabular Patient Data

### File Format
- **Format**: CSV (Comma-Separated Values)
- **Encoding**: UTF-8
- **File Name**: `sample_patient_data.csv`
- **Delimiter**: Comma (`,`)
- **Header**: Required (first row)

### Schema

| Column Name | Data Type | Description | Valid Range | Missing Value Handling |
|------------|-----------|-------------|-------------|----------------------|
| `patient_id` | String | Unique patient identifier | Format: `P###` (e.g., P001) | **Not Allowed** - Primary key |
| `age` | Integer | Patient age in years | 18-120 | Impute with **median age** from dataset |
| `sex` | Categorical | Biological sex | `M` (Male), `F` (Female) | Impute with **mode** or create `Unknown` category |
| `bmi` | Float | Body Mass Index | 15.0-50.0 | Impute with **mean BMI**, flag outliers |
| `bp` | String | Blood pressure reading | Format: `###/##` (systolic/diastolic) | Impute with **median systolic/diastolic** separately |
| `chol` | Integer | Total cholesterol (mg/dL) | 100-400 | Impute with **median cholesterol** |
| `glucose` | Integer | Fasting glucose (mg/dL) | 70-300 | Impute with **median glucose** |
| `label` | Binary | Risk classification | `0` (Low Risk), `1` (High Risk) | **Not Allowed** - Target variable |

### Data Validation Rules
1. **Age**: Must be positive integer. Values outside 18-120 flagged as outliers.
2. **BMI**: Calculated as weight(kg) / height(m)². Normal range: 18.5-30.0.
3. **Blood Pressure**: Parse systolic/diastolic separately for analysis. Format validation: `###/##`.
4. **Cholesterol**: Total cholesterol. Desirable: <200, Borderline: 200-239, High: ≥240.
5. **Glucose**: Fasting glucose. Normal: <100, Prediabetes: 100-125, Diabetes: ≥126.

### Missing Value Strategy
- **Numerical Features** (`age`, `bmi`, `chol`, `glucose`): Impute with **median** of non-missing values.
- **Blood Pressure**: Split into systolic/diastolic, impute each separately with **median**.
- **Categorical Features** (`sex`): Impute with **mode** or create dedicated `Unknown` category.
- **Critical Fields** (`patient_id`, `label`): Rows with missing values **must be excluded** from training.

### Example Data
```csv
patient_id,age,sex,bmi,bp,chol,glucose,label
P001,45,M,28.5,130/85,195,102,0
P002,62,F,31.2,145/92,220,128,1
```

---

## 2. Radiology Images

### File Format
- **Format**: PNG (Portable Network Graphics)
- **Color Space**: Grayscale (8-bit)
- **Dimensions**: 512×512 pixels (standardized)
- **File Naming**: `sample_radiology_image.png` or `{patient_id}_radiology.png`

### Preprocessing Requirements
1. **Normalization**: Scale pixel values to [0, 1] range by dividing by 255.
2. **Resizing**: Resize all images to 512×512 if dimensions vary.
3. **Enhancement**: Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) if needed.
4. **Augmentation** (Training only): Random rotation (±10°), horizontal flip, brightness/contrast adjustment.

### Missing Value Handling
- **Missing Images**: If radiology image is unavailable for a patient:
  - Use **zero-filled tensor** of shape (512, 512, 1) as placeholder.
  - Set `has_image` flag to `False` in metadata.
  - Optionally exclude patient from multi-modal models requiring images.

### Expected Image Content
- Chest X-rays (frontal PA view)
- Lung fields, heart shadow, and diaphragm should be visible
- Medical annotations/markers acceptable but not required

---

## 3. Medical Report Text

### File Format
- **Format**: Plain text file (.txt)
- **Encoding**: UTF-8
- **Structure**: Patient ID followed by colon, then free-text report
- **File Name**: `sample_medical_reports.txt`

### Schema
```
{patient_id}: {free_text_medical_report}
```

### Example Entry
```
P001: Patient presents with mild hypertension and slightly elevated cholesterol. 
BMI within normal range. Recommend lifestyle modifications including diet and 
regular exercise. No significant cardiovascular abnormalities detected on ECG.
```

### Text Preprocessing
1. **Tokenization**: Use medical domain-specific tokenizer (e.g., ClinicalBERT, BioClinicalBERT).
2. **Length**: Truncate or pad to maximum sequence length (512 tokens recommended).
3. **Cleaning**: 
   - Remove excessive whitespace
   - Preserve medical abbreviations (do not expand)
   - Normalize line breaks to single space
4. **Embeddings**: Generate embeddings using pretrained clinical language model.

### Missing Value Handling
- **Missing Reports**: If text report unavailable:
  - Use **empty string** or special token `[NO_REPORT]`.
  - Generate **zero embedding vector** of appropriate dimension.
  - Set `has_report` flag to `False` in metadata.

---

## 4. Graph Construction Rules

MED-RiskNET constructs heterogeneous graphs to capture relationships between patients, features, and medical entities. Below are the rules for graph construction.

### Node Types

| Node Type | Description | Features | Source |
|-----------|-------------|----------|--------|
| **Patient** | Individual patient | All tabular features + embeddings | Tabular CSV |
| **Feature** | Discrete feature values (e.g., age groups, BMI categories) | One-hot encoding | Derived from tabular data |
| **Image** | Radiology scan representations | CNN embeddings (ResNet/Vision Transformer) | Radiology images |
| **Report** | Medical report text | Language model embeddings (BERT/BioClinicalBERT) | Medical reports |

### Edge Types

| Edge Type | Source → Target | Description | Construction Rule |
|-----------|----------------|-------------|-------------------|
| **has_feature** | Patient → Feature | Patient exhibits feature | Connect patient to discretized feature nodes (e.g., age_group_40-50) |
| **has_image** | Patient → Image | Patient has radiology scan | Connect patient to their image embedding node |
| **has_report** | Patient → Report | Patient has medical report | Connect patient to their report embedding node |
| **similar_to** | Patient → Patient | Clinical similarity | Connect patients with similar risk profiles (cosine similarity > 0.8) |
| **feature_correlation** | Feature → Feature | Statistical correlation | Connect features with Pearson correlation > 0.6 |

### Graph Construction Algorithm

#### Step 1: Create Patient Nodes
```python
for each row in tabular_data:
    create patient_node with:
        - patient_id (unique identifier)
        - age, sex, bmi, bp, chol, glucose (continuous features)
        - label (target variable)
```

#### Step 2: Create Feature Nodes (Discretization)
```python
# Example: Age binning
age_bins = [18, 30, 40, 50, 60, 70, 120]
age_groups = pd.cut(age, bins=age_bins, labels=['18-30', '30-40', '40-50', '50-60', '60-70', '70+'])

# Create feature nodes
for each unique value in age_groups, sex, bmi_category, etc.:
    create feature_node
```

#### Step 3: Create Multi-Modal Nodes
```python
# Image nodes
for each patient with radiology image:
    image_embedding = CNN_model.extract_features(image)
    create image_node with embedding

# Report nodes
for each patient with medical report:
    report_embedding = BERT_model.encode(report_text)
    create report_node with embedding
```

#### Step 4: Add Edges
```python
# Patient-Feature edges
for each patient:
    for each discretized feature:
        add_edge(patient_node, feature_node, type='has_feature')

# Patient-Image edges
for each patient with image:
    add_edge(patient_node, image_node, type='has_image')

# Patient-Report edges
for each patient with report:
    add_edge(patient_node, report_node, type='has_report')

# Patient-Patient similarity edges
for each pair of patients (i, j):
    similarity = cosine_similarity(patient_i_features, patient_j_features)
    if similarity > threshold (e.g., 0.8):
        add_edge(patient_i, patient_j, type='similar_to', weight=similarity)

# Feature-Feature correlation edges
correlation_matrix = compute_correlation(all_features)
for each pair of features (f1, f2):
    if abs(correlation[f1, f2]) > threshold (e.g., 0.6):
        add_edge(f1, f2, type='feature_correlation', weight=correlation)
```

### Graph Statistics Example
For 10 sample patients:
- **Nodes**: ~10 patient nodes + ~15 feature nodes + ~10 image nodes + ~10 report nodes = **~45 nodes**
- **Edges**: 
  - Patient-Feature: ~50 edges (each patient has ~5 discretized features)
  - Patient-Image: ~10 edges
  - Patient-Report: ~10 edges
  - Patient-Patient: ~5-10 edges (similarity-based)
  - Feature-Feature: ~8-12 edges (correlation-based)
  - **Total**: ~80-90 edges

### Graph Representation Format
- **Library**: PyTorch Geometric (PyG) or DGL
- **Storage**: 
  - Node features: Dense tensor of shape `[num_nodes, feature_dim]`
  - Edge indices: COO format `[2, num_edges]`
  - Edge attributes: Optional weights, types
- **File Format**: `.pt` (PyTorch) or `.bin` (DGL binary format)

---

## 5. Data Quality Checks

### Automated Validation Checklist
- [ ] All `patient_id` values are unique
- [ ] No missing values in `patient_id` or `label` columns
- [ ] Age values within valid range [18, 120]
- [ ] BMI values within valid range [15.0, 50.0]
- [ ] Blood pressure format matches `###/##` pattern
- [ ] Sex values are either `M` or `F` (or missing)
- [ ] Label values are binary (0 or 1)
- [ ] All referenced images exist in file system
- [ ] All images are grayscale PNG format
- [ ] All medical reports are properly keyed by `patient_id`
- [ ] No duplicate `patient_id` entries in reports

### Data Loading Pipeline
```python
import pandas as pd
import numpy as np
from PIL import Image

# Load tabular data
df = pd.read_csv('sample_patient_data.csv')

# Validate schema
assert 'patient_id' in df.columns
assert df['patient_id'].is_unique
assert df['label'].isin([0, 1]).all()

# Handle missing values
df['age'].fillna(df['age'].median(), inplace=True)
df['bmi'].fillna(df['bmi'].mean(), inplace=True)
df['sex'].fillna(df['sex'].mode()[0], inplace=True)

# Parse blood pressure
df[['bp_systolic', 'bp_diastolic']] = df['bp'].str.split('/', expand=True).astype(int)

# Load images
for patient_id in df['patient_id']:
    img_path = f'data/{patient_id}_radiology.png'
    if os.path.exists(img_path):
        img = Image.open(img_path).convert('L')
        img = img.resize((512, 512))
    else:
        img = np.zeros((512, 512), dtype=np.uint8)
    
# Load reports
with open('sample_medical_reports.txt', 'r') as f:
    reports = {}
    for line in f:
        if ':' in line:
            pid, report = line.split(':', 1)
            reports[pid.strip()] = report.strip()
```

---

## 6. Versioning and Updates

- **Version**: 1.0
- **Last Updated**: 2025-12-02
- **Changelog**:
  - v1.0: Initial data contract with sample datasets

### Contact
For questions or updates to this data contract, please contact the MED-RiskNET data team.

---

## 7. References

- **Clinical Guidelines**: AHA/ACC Cardiovascular Risk Assessment
- **Data Standards**: HL7 FHIR, DICOM for medical imaging
- **Privacy**: HIPAA-compliant data handling (all sample data is synthetic)
