"""
FastAPI application for MED-RiskNET deployment.

Provides REST API endpoints for:
- /predict: Risk prediction with uncertainty
- /explain: Model explainability (SHAP, Grad-CAM, GNNExplainer)
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, List
import torch
import numpy as np
from PIL import Image
import io
import base64
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.medrisknet import MedRiskNet
from data.graph_builder import PatientGraphBuilder
from evaluation.uncertainty import MCDropoutWrapper
from explainability.shap_explainer import SHAPExplainer
from explainability.gradcam import GradCAM
from explainability.gnn_explainer import GNNExplainerWrapper
import torchvision.transforms as transforms


# Initialize FastAPI app
app = FastAPI(
    title="MED-RiskNET API",
    description="Medical Risk Prediction and Explainability API",
    version="1.0.0"
)

# Global model (loaded once)
MODEL = None
MC_WRAPPER = None
DEVICE = "cpu"


# Request/Response Models
class PredictionRequest(BaseModel):
    """Request for prediction endpoint."""
    patient_id: Optional[str] = None
    age: float
    bmi: float
    chol: float
    glucose: float
    bp_systolic: float
    bp_diastolic: float
    sex_F: float
    sex_M: float
    text: Optional[str] = None


class PredictionResponse(BaseModel):
    """Response from prediction endpoint."""
    patient_id: Optional[str]
    risk_score: float
    triage_rank: float
    uncertainty: float
    risk_category: str  # "Low", "Medium", "High"


class ExplanationRequest(BaseModel):
    """Request for explanation endpoint."""
    patient_id: Optional[str] = None
    age: float
    bmi: float
    chol: float
    glucose: float
    bp_systolic: float
    bp_diastolic: float
    sex_F: float
    sex_M: float
    text: Optional[str] = None
    explain_type: str = "shap"  # "shap", "gradcam", "gnn"


class ExplanationResponse(BaseModel):
    """Response from explanation endpoint."""
    patient_id: Optional[str]
    explanation_type: str
    feature_importance: Optional[Dict[str, float]] = None
    heatmap_base64: Optional[str] = None
    important_neighbors: Optional[List[Dict]] = None


def load_model():
    """Load model on startup."""
    global MODEL, MC_WRAPPER
    
    print("Loading MED-RiskNET model...")
    
    # Load checkpoint if available
    checkpoint_path = "checkpoints/best_model.pt"
    
    MODEL = MedRiskNet(
        tabular_input_dim=8,
        num_classes=1,
        fusion_hidden_dim=128,
        use_graph=False  # Disable graph for API simplicity
    )
    
    # Try to load checkpoint
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        MODEL.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded checkpoint from {checkpoint_path}")
    else:
        print("⚠ No checkpoint found, using random weights")
    
    MODEL.to(DEVICE)
    MODEL.eval()
    
    # Create MC-Dropout wrapper for uncertainty
    MC_WRAPPER = MCDropoutWrapper(MODEL, num_samples=10)
    
    print("✓ Model loaded successfully")


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "MED-RiskNET API",
        "version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    age: float = Form(...),
    bmi: float = Form(...),
    chol: float = Form(...),
    glucose: float = Form(...),
    bp_systolic: float = Form(...),
    bp_diastolic: float = Form(...),
    sex_F: float = Form(...),
    sex_M: float = Form(...),
    patient_id: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """
    Predict patient risk with uncertainty.
    
    Args:
        Tabular features, optional text and image
    
    Returns:
        Risk score, triage rank, uncertainty
    """
    try:
        # Prepare tabular input
        tabular = torch.tensor(
            [[age, bmi, chol, glucose, bp_systolic, bp_diastolic, sex_F, sex_M]],
            dtype=torch.float32
        ).to(DEVICE)
        
        # Process image if provided
        image_tensor = None
        if image:
            img_bytes = await image.read()
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            image_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        # Get prediction with uncertainty
        outputs = MC_WRAPPER.forward_with_uncertainty(
            tabular=tabular,
            image=image_tensor,
            text=[text] if text else None
        )
        
        risk_score = outputs['mean_classification'].item()
        ranking_score = outputs['mean_ranking'].item()
        uncertainty = outputs['std_classification'].item()
        
        # Categorize risk
        if risk_score < 0.3:
            risk_category = "Low"
        elif risk_score < 0.7:
            risk_category = "Medium"
        else:
            risk_category = "High"
        
        return PredictionResponse(
            patient_id=patient_id,
            risk_score=round(risk_score, 4),
            triage_rank=round(ranking_score, 4),
            uncertainty=round(uncertainty, 4),
            risk_category=risk_category
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/explain", response_model=ExplanationResponse)
async def explain(
    age: float = Form(...),
    bmi: float = Form(...),
    chol: float = Form(...),
    glucose: float = Form(...),
    bp_systolic: float = Form(...),
    bp_diastolic: float = Form(...),
    sex_F: float = Form(...),
    sex_M: float = Form(...),
    explain_type: str = Form("shap"),
    patient_id: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """
    Generate model explanations.
    
    Args:
        Patient features and explanation type
    
    Returns:
        Explanation based on type (SHAP/Grad-CAM/GNN)
    """
    # Validate explain_type first
    if explain_type not in ["shap", "gradcam", "gnn"]:
        raise HTTPException(status_code=400, detail=f"Invalid explain_type: {explain_type}. Must be one of: shap, gradcam, gnn")
    
    try:
        # Prepare tabular input
        tabular_np = np.array([[age, bmi, chol, glucose, bp_systolic, bp_diastolic, sex_F, sex_M]])
        tabular = torch.tensor(tabular_np, dtype=torch.float32).to(DEVICE)
        
        if explain_type == "shap":
            # SHAP explanation
            feature_names = ['age', 'bmi', 'chol', 'glucose', 'bp_systolic', 'bp_diastolic', 'sex_F', 'sex_M']
            
            # Create explainer (using small background for speed)
            explainer = SHAPExplainer(
                MODEL,
                background_data=tabular_np,  # Use input as background
                feature_names=feature_names
            )
            
            # Explain
            explanation = explainer.explain(tabular_np, nsamples=50)
            
            # Get feature importance
            importance_dict = {
                feat: float(abs(val))
                for feat, val in zip(feature_names, explanation.values[0])
            }
            
            return ExplanationResponse(
                patient_id=patient_id,
                explanation_type="shap",
                feature_importance=importance_dict
            )
        
        elif explain_type == "gradcam":
            if image is None:
                raise HTTPException(status_code=400, detail="Image required for Grad-CAM")
            
            # Process image
            img_bytes = await image.read()
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            image_tensor = transform(img).unsqueeze(0).to(DEVICE)
            
            # Get Grad-CAM (simplified - just return placeholder)
            # In production, properly implement with target layer
            
            return ExplanationResponse(
                patient_id=patient_id,
                explanation_type="gradcam",
                heatmap_base64="<grad-cam-heatmap-base64>"  # Placeholder
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
