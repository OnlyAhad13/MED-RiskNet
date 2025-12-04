# MED-RiskNET: Multimodal Medical Risk Prediction Network

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A state-of-the-art multimodal deep learning system for medical risk prediction and patient triage.**

MED-RiskNET integrates **tabular** clinical data, **medical images**, **clinical text**, and **patient similarity graphs** to predict health risks with uncertainty quantification and full explainability.

---

## 🏗️ **Architecture**

```
┌─────────────────────────────────────────────────────────────────────┐
│                      MED-RiskNET ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────────┘

INPUT MODALITIES:
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Tabular    │  │    Image     │  │     Text     │  │    Graph     │
│ (age, BP,    │  │  (Chest      │  │  (Clinical   │  │  (Patient    │
│  BMI, etc.)  │  │   X-rays)    │  │   Reports)   │  │  Similarity) │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │                 │
       ▼                 ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Tabular    │  │     CNN      │  │BioClinical   │  │  GraphSAGE   │
│   Encoder    │  │  (ResNet50)  │  │    BERT      │  │   + GAT      │
│  [8 → 32]    │  │ [2048 → 512] │  │ [768]        │  │  [8 → 16]    │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │                 │
       └─────────────────┴─────────────────┴─────────────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │  FUSION TRANSFORMER   │
                  │  (Multi-Head Self-    │
                  │   Attention Fusion)   │
                  │    [All → 256]        │
                  └───────────┬───────────┘
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
    ┌───────────────────┐      ┌───────────────────┐
    │ Classification    │      │    Ranking        │
    │     Head          │      │     Head          │
    │  (Risk Prob)      │      │  (Triage Score)   │
    └─────────┬─────────┘      └─────────┬─────────┘
              │                          │
              ▼                          ▼
    ┌─────────────────────────────────────────┐
    │         DUAL PREDICTIONS                │
    │  • Risk Probability [0-1]               │
    │  • Ranking Score (for triage)           │
    │  • Uncertainty (MC-Dropout)             │
    └─────────────────────────────────────────┘

LOSS FUNCTION:
Combined Loss = BCE(classification) + λ × PairwiseHinge(ranking)
```

---

## 📊 **Key Features**

### **1. Multimodal Integration**
- **Tabular**: Clinical vitals (age, BMI, blood pressure, cholesterol, glucose)
- **Image**: Chest X-rays via pretrained ResNet50
- **Text**: Clinical reports via BioClinicalBERT
- **Graph**: Patient similarity networks via GraphSAGE/GAT

### **2. Dual Task Learning**
- **Classification**: Binary risk prediction with calibrated probabilities
- **Ranking**: Pairwise patient ordering for triage prioritization

### **3. Uncertainty Quantification**
- MC-Dropout for prediction confidence intervals
- Temperature scaling for probability calibration

### **4. Full Explainability**
- **SHAP**: Tabular feature importance
- **Grad-CAM**: Visual attention heatmaps
- **GNNExplainer**: Important patient neighbors

### **5. Production-Ready API**
- FastAPI REST endpoints
- Docker containerization
- Comprehensive testing

---

## 🚀 **Quick Start**

### **Installation**

```bash
# Clone repository
git clone https://github.com/yourusername/MED-RiskNET
cd MED-RiskNET

# Create conda environment
conda create -n medrisknet python=3.11
conda activate medrisknet

# Install dependencies
pip install -r requirements.txt

# Install optional dependencies
pip install shap  # For SHAP explainability
pip install torch-geometric  # For graph neural networks
```

### **Dataset Preparation**

#### **1. Tabular Data** (`data/patient_data.csv`)
```csv
patient_id,age,sex,bmi,bp,chol,glucose,label
P001,65,M,28.5,140/90,240,120,1
P002,45,F,22.0,120/80,180,95,0
...
```

#### **2. Medical Images** (`data/images/`)
```
data/images/
├── P001_radiology.png
├── P002_radiology.png
└── ...
```

#### **3. Clinical Reports** (`data/medical_reports.txt`)
```
P001: Patient presents with hypertension and elevated cholesterol...
P002: Normal vital signs, no significant findings...
```

---

## 🎯 **Training**

### **Basic Training**

```bash
# Train for 1 epoch (smoke test)
python tests/test_train_smoke.py

# Full training
python training/train.py \
    --data data/patient_data.csv \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.001 \
    --checkpoint-dir checkpoints/
```

### **Training Configuration**

```python
# Key hyperparameters
TABULAR_DIM = 8         # Input features
FUSION_DIM = 256        # Fusion representation
NUM_CLASSES = 1         # Binary classification
RANKING_LOSS_WEIGHT = 0.5
LEARNING_RATE = 1e-3
SCHEDULER = "CosineAnnealing"
```

---

## 📈 **Evaluation**

### **Comprehensive Metrics**

```bash
python evaluation/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --data-csv data/patient_data.csv \
    --image-dir data/images \
    --text-file data/medical_reports.txt
```

**Output:**
```
Classification Metrics:
  - AUROC:     0.8523
  - AUPRC:     0.7891
  - Precision: 0.8123
  - Recall:    0.7654
  - F1 Score:  0.7882

Ranking Metrics:
  - Recall@10: 0.8333
  - NDCG:      0.8912

Calibration:
  - ECE:       0.0421

Uncertainty:
  - Mean:      0.0523
```

---

## 🔬 **Ablation Studies**

### **Template: Modality Ablation**

```bash
# Baseline (all modalities)
python training/train.py --use-all-modalities

# Without images
python training/train.py --no-image

# Without text
python training/train.py --no-text

# Without graph
python training/train.py --no-graph

# Tabular only
python training/train.py --tabular-only
```

### **Template: Architecture Ablation**

```python
# models/config.py

# Baseline: Transformer Fusion
FUSION_TYPE = "transformer"

# Alternative: Cross-Modal Attention
FUSION_TYPE = "cross_attention"

# Alternative: Simple Concatenation
FUSION_TYPE = "concat"
```

### **Template: Loss Ablation**

```bash
# Classification only
python training/train.py --ranking-weight 0.0

# Ranking only
python training/train.py --ranking-weight 1.0

# Balanced (default)
python training/train.py --ranking-weight 0.5
```

---

## 🌐 **API Deployment**

### **Start Server**

```bash
# Local development
bash scripts/serve.sh

# Production (Docker)
docker build -t medrisknet:latest .
docker run -p 8000:8000 medrisknet:latest
```

### **API Usage**

```python
import requests

# Predict patient risk
response = requests.post("http://localhost:8000/predict", data={
    "age": 65.0,
    "bmi": 28.5,
    "chol": 240.0,
    "glucose": 120.0,
    "bp_systolic": 140.0,
    "bp_diastolic": 90.0,
    "sex_F": 0.0,
    "sex_M": 1.0,
    "text": "Patient with hypertension"
})

result = response.json()
# {
#   "risk_score": 0.87,
#   "triage_rank": 2.45,
#   "uncertainty": 0.05,
#   "risk_category": "High"
# }

# Get SHAP explanation
response = requests.post("http://localhost:8000/explain", data={
    ...,  # Same features
    "explain_type": "shap"
})

explanation = response.json()
# {
#   "feature_importance": {
#     "glucose": 0.002,
#     "bp_systolic": 0.0014,
#     ...
#   }
# }
```

---

## 🔍 **Explainability**

### **SHAP (Tabular)**

```bash
python explainability/examples.py
```

Generates:
- `shap_waterfall.png` - Individual prediction explanation
- `shap_summary.png` - Global feature importance

### **Grad-CAM (Images)**

```python
from explainability.gradcam import GradCAM

gradcam = GradCAM(model.image_encoder, target_layer)
heatmap = gradcam.generate_cam(image)
# Visualizes which image regions influence prediction
```

### **GNNExplainer (Graph)**

```python
from explainability.gnn_explainer import GNNExplainerWrapper

explainer = GNNExplainerWrapper(model.graph_encoder)
explanation = explainer.explain_node(patient_idx, graph_data)
# Identifies which similar patients influence prediction
```

---

## 📝 **Reproducing Results**

### **1. Environment Setup**

```bash
conda env create -f environment.yml
conda activate medrisknet
```

### **2. Data Preparation**

```bash
# Organize your data following the structure:
data/
├── patient_data.csv
├── images/
│   ├── P001_radiology.png
│   └── ...
└── medical_reports.txt
```

### **3. Training**

```bash
# Set random seed for reproducibility
python training/train.py --seed 42 --epochs 50
```

### **4. Evaluation**

```bash
python evaluation/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --seed 42
```

### **5. Expected Results**

| Metric | Expected Value |
|--------|---------------|
| AUROC | 0.85 ± 0.02 |
| AUPRC | 0.78 ± 0.03 |
| NDCG | 0.89 ± 0.01 |
| ECE | < 0.05 |

---

## 📚 **Citations**

If you use MED-RiskNET in your research, please cite:

```bibtex
@article{medrisknet2025,
  title={MED-RiskNET: Multimodal Medical Risk Prediction with Explainability},
  author={Syed Abdul Ahad},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2025}
}
```

---

## 🤝 **Contributing**

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 **License**

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## ⚠️ **Medical Disclaimer**

**This software is for research purposes only and is NOT intended for clinical use.** All predictions should be reviewed by qualified healthcare professionals. See [MODEL_CARD.md](MODEL_CARD.md) for detailed limitations and risk considerations.

---

## 📞 **Contact**

- **Email**: syedahad171@gmail.com
- **Issues**: [GitHub Issues](https://github.com/OnlyAhad13/MED-RiskNet/issues)
- **Discussions**: [GitHub Discussions](https://github.com/OnlyAhad13/MED-RiskNet/discussions)

---

## 🙏 **Acknowledgments**

- BioClinicalBERT: Alsentzer et al.
- PyTorch Geometric: Fey & Lenssen
- SHAP: Lundberg & Lee
- Grad-CAM: Selvaraju et al.
