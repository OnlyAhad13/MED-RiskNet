
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import argparse
import json
from models.medrisknet import MedRiskNet
from evaluation.uncertainty import MCDropoutWrapper

def main():
    parser = argparse.ArgumentParser(description="Run inference on a single patient.")
    parser.add_argument("--checkpoint", type=str, default="test_checkpoints/best_model.pt")
    # Add arguments for features
    parser.add_argument("--age", type=float, default=65.0)
    parser.add_argument("--bmi", type=float, default=28.5)
    parser.add_argument("--chol", type=float, default=240.0)
    parser.add_argument("--glucose", type=float, default=120.0)
    parser.add_argument("--bp-systolic", type=float, default=140.0)
    parser.add_argument("--bp-diastolic", type=float, default=90.0)
    parser.add_argument("--sex-f", type=float, default=0.0)
    parser.add_argument("--sex-m", type=float, default=1.0)
    parser.add_argument("--text", type=str, default="Patient has history of hypertension.")
    
    args = parser.parse_args()

    # Load Model
    print(f"Loading model from {args.checkpoint}...", file=sys.stderr)
    try:
        model = MedRiskNet(
            tabular_input_dim=8,
            num_classes=1,
            fusion_hidden_dim=128,
            use_graph=True  # Enabled to match checkpoint keys
        )
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    # Wrap for uncertainty
    mc_model = MCDropoutWrapper(model, num_samples=10)

    # Prepare Input
    tabular = torch.tensor([[
        args.age, args.bmi, args.chol, args.glucose, 
        args.bp_systolic, args.bp_diastolic, args.sex_f, args.sex_m
    ]], dtype=torch.float32)

    # Inference
    print("Running inference...", file=sys.stderr)
    with torch.no_grad():
        outputs = mc_model.forward_with_uncertainty(
            tabular=tabular,
            text=[args.text],
            image=None # Skipping image for simple CLI demo
        )

    # Format Output
    risk_score = outputs['mean_classification'].item()
    rank_score = outputs['mean_ranking'].item()
    uncertainty = outputs['std_classification'].item()

    if risk_score < 0.3:
        category = "Low"
    elif risk_score < 0.7:
        category = "Medium"
    else:
        category = "High"

    result = {
        "risk_probability": round(risk_score, 4),
        "risk_category": category,
        "triage_score": round(rank_score, 4),
        "uncertainty_score": round(uncertainty, 4),
        "input_features": {
            "age": args.age,
            "bp": f"{int(args.bp_systolic)}/{int(args.bp_diastolic)}",
            "text": args.text
        }
    }

    # Print JSON
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
