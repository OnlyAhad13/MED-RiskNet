
<div align="center">

# 🏥 MED-RiskNET
### Multimodal Medical Risk Prediction Network

**A state-of-the-art deep learning system that integrates Clinical Records, Medical Imaging, Doctor's Notes, and Patient Similarity Graphs.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/Transformers-4.30-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![PyG](https://img.shields.io/badge/PyG-Graph_Neural_Nets-3C2179?style=for-the-badge&logo=graph&logoColor=white)](https://pytorch-geometric.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-Deployment-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

[View Architecture](#-architecture) • [Key Features](#-key-features) • [Quick Start](#-quick-start) • [Inference Demo](#-inference-demo)

</div>

---

## 🏗️ **Architecture**

MED-RiskNET uses a **Late-Fusion Architecture** to process four distinct data modalities simultaneously.

```mermaid
graph TD
    subgraph Inputs
    T[Tabular Vitals] --> E1[MLP Encoder]
    I[Chest X-Rays] --> E2[ResNet50]
    Txt[Clinical Notes] --> E3[BioClinicalBERT]
    G[Patient Graph] --> E4[GraphSAGE + GAT]
    end

    subgraph Fusion
    E1 --> F[Fusion Transformer]
    E2 --> F
    E3 --> F
    E4 --> F
    end

    subgraph "Dual Outputs"
    F --> C[Risk Probability]
    F --> R[Triage Ranking]
    end
    
    style F fill:#f9f,stroke:#333,stroke-width:4px
    style C fill:#bbf,stroke:#333,stroke-width:2px
    style R fill:#bbf,stroke:#333,stroke-width:2px
```

---

## 📊 **Key Features**

| Feature | Description | Tech Stack |
| :--- | :--- | :--- |
| **Multimodal Integration** | Fuses Vitals, X-Rays, Text, and Graph structure. | `TorchFusion`, `Transformer` |
| **Dual-Task Learning** | Predicts absolute **Risk** and relative **Triage Rank** simultaneously. | `BCE Loss` + `Pairwise Hinge Loss` |
| **Uncertainty Quantification** | Knows when it doesn't know. Outputs confidence scores. | `MC-Dropout` |
| **Full Explainability** | Tells you *why*: Feature importance, Image Heatmaps, Graph Neighbors. | `SHAP`, `Grad-CAM`, `GNNExplainer` |
| **Production Ready** | REST API with documentation and Docker support. | `FastAPI`, `Docker` |

---

## 🚀 **Quick Start**

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/OnlyAhad13/MED-RiskNet.git
cd MED-RiskNet

# Create environment
conda create -n medrisknet python=3.11
conda activate medrisknet

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Inference (New!)

We provide a CLI tool to run the model on a single simulated patient instantly.

```bash
python scripts/infer.py
```

**Sample Output:**
```json
{
  "risk_probability": 0.4921,
  "risk_category": "Medium",
  "triage_score": -0.0812,
  "uncertainty_score": 0.0281,
  "input_features": {
    "age": 65,
    "bp": "140/90",
    "text": "Patient has history of hypertension."
  }
}
```

### 3. Training & Evaluation

<details>
<summary><b>Click to expand Training commands</b></summary>

```bash
# Run a quick smoke test
python tests/test_train_smoke.py

# Full Evaluation
python evaluation/evaluate.py --checkpoint test_checkpoints/best_model.pt
```
</details>

---

## 🌐 **API Deployment**

Get a production-ready REST API up and running in seconds.

```bash
# Start the server locally
bash scripts/serve.sh
```

**Docs are available at:** `http://localhost:8000/docs`

---

## 🔍 **Explainability Gallery**

| SHAP (Tabular) | Grad-CAM (Image) |
| :---: | :---: |
| *Identifies key physiological factors* | *Highlights suspicious regions in X-Ray* |
| `explainability/shap_explainer.py` | `explainability/gradcam.py` |

---

## 📚 **Citations**

If you use **MED-RiskNET**, please cite:

```bibtex
@article{medrisknet2025,
  title={MED-RiskNET: Multimodal Medical Risk Prediction with Explainability},
  author={Syed Abdul Ahad},
  year={2025}
}
```

<div align="center">

**[🐛 Report Bug](https://github.com/OnlyAhad13/MED-RiskNet/issues) • [📝 Request Feature](https://github.com/OnlyAhad13/MED-RiskNet/issues)**

Made with ❤️ by [OnlyAhad13](https://github.com/OnlyAhad13)

</div>
