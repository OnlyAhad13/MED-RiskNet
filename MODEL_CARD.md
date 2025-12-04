# Model Card: MED-RiskNET

## Model Details

- **Name:** MED-RiskNET (Multimodal Medical Risk Prediction Network)
- **Version:** 1.0.0
- **Date:** December 2025
- **Model Type:** Multimodal Deep Neural Network (Tabular + Image + Text + Graph)
- **Architecture:** 
  - **Tabular:** Residual MLP
  - **Image:** ResNet50 (Pretrained on ImageNet)
  - **Text:** BioClinicalBERT (Pretrained)
  - **Graph:** GraphSAGE + GAT (Graph Attention Network)
  - **Fusion:** Transformer-based Multi-Head Attention
- **License:** MIT License

---

## Intended Use

### **Primary Use Cases**
1. **Risk Stratification:** Predicting the probability of high-risk medical events (e.g., readmission, mortality, disease progression) for hospitalized patients.
2. **Patient Triage:** Ranking patients based on severity to prioritize care resources in emergency or ICU settings.
3. **Clinical Decision Support:** Providing explainable risk scores to assist clinicians in making informed decisions.

### **Target Users**
- Clinical Researchers
- Healthcare Data Scientists
- Hospital Administrators (for resource planning)
- Clinicians (as a support tool, NOT a replacement)

### **Out of Scope Use Cases**
- **Autonomous Diagnosis:** The model should NOT be used to diagnose conditions without human oversight.
- **Treatment Prescription:** The model does NOT recommend specific treatments.
- **Pediatric Populations:** The model was trained on adult data and may not generalize to children.
- **Real-time Critical Care:** Latency may not be sufficient for sub-second critical care decisions.

---

## Performance Metrics

The model is evaluated on a hold-out test set using the following metrics:

| Metric | Description | Target Performance |
|--------|-------------|-------------------|
| **AUROC** | Area Under Receiver Operating Characteristic Curve | > 0.85 |
| **AUPRC** | Area Under Precision-Recall Curve | > 0.75 |
| **Recall@K** | Proportion of true high-risk patients in top K predictions | > 0.80 (at K=10) |
| **NDCG** | Normalized Discounted Cumulative Gain (Ranking Quality) | > 0.85 |
| **ECE** | Expected Calibration Error (Probability Accuracy) | < 0.05 |

---

## Limitations & Biases
### **1. Demographic Bias**
- **Representation:** If the training data is imbalanced regarding age, gender, or ethnicity, the model may exhibit performance disparities across these groups.
- **Fairness Checks:** We recommend running fairness audits (e.g., equalized odds, demographic parity) before deployment.

### **2. Generalizability**
- **Domain Shift:** The model may not generalize well to hospitals with different data collection protocols or patient demographics than the training site.

---

## Risk Mitigation Strategies

### **1. Uncertainty Quantification**
- The model outputs an **uncertainty score** (via MC-Dropout) alongside the risk prediction.
- **Action:** Predictions with high uncertainty (> threshold) should be flagged for manual review.

### **2. Explainability**
- **SHAP Values:** Provide feature importance for tabular data.
- **Grad-CAM:** Highlights relevant regions in medical images.
- **Action:** Clinicians should review these explanations to verify the model's reasoning aligns with medical knowledge.

### **3. Human-in-the-Loop**
- The model is designed as a **Decision Support System**.
- **Action:** Final decisions must always be made by qualified healthcare professionals.

---

## Ethical Considerations

- **Privacy:** Patient data must be de-identified before processing. The model does not store PII.
- **Transparency:** The model architecture and training process are open-source to ensure transparency.
- **Accountability:** Clear logs of model predictions and versions should be maintained for audit trails.

---

## Training Data

- **Source:** [HAIM-Multimodal Dataset]
- **Preprocessing:**
  - Tabular: Standardization, One-Hot Encoding, Median Imputation
  - Image: Resize to 224x224, Normalization (ImageNet stats)
  - Text: Tokenization (BioClinicalBERT tokenizer)
  - Graph: k-NN graph construction based on tabular similarity

---

## How to Get Started

Refer to the [README.md](README.md) for installation, training, and evaluation instructions.

```bash
python evaluation/evaluate.py --checkpoint checkpoints/best_model.pt
```
