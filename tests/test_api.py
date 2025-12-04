"""
API endpoint tests for MED-RiskNET deployment.

Tests:
- GET / (health check)
- POST /predict (risk prediction)
- POST /explain (explainability)

Usage:
    # Start server first:
    bash scripts/serve.sh
    
    # Then run tests:
    pytest tests/test_api.py
"""

import requests
import pytest
from pathlib import Path
import time


# API base URL
BASE_URL = "http://localhost:8000"


@pytest.fixture(scope="module")
def wait_for_server():
    """Wait for server to be ready."""
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                print("✓ Server is ready")
                return
        except requests.ConnectionError:
            if i == 0:
                print("Waiting for server to start...")
            time.sleep(1)
    
    pytest.fail("Server did not start in time")


def test_root_endpoint(wait_for_server):
    """Test root endpoint."""
    response = requests.get(f"{BASE_URL}/")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "status" in data
    assert data["status"] == "healthy"
    assert "service" in data
    assert "version" in data
    
    print("✓ Root endpoint test passed")


def test_health_endpoint(wait_for_server):
    """Test health endpoint."""
    response = requests.get(f"{BASE_URL}/health")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "status" in data
    assert "model_loaded" in data
    assert data["model_loaded"] is True
    
    print("✓ Health endpoint test passed")


def test_predict_endpoint(wait_for_server):
    """Test prediction endpoint."""
    # Sample patient data
    data = {
        "patient_id": "TEST001",
        "age": 65.0,
        "bmi": 28.5,
        "chol": 240.0,
        "glucose": 120.0,
        "bp_systolic": 140.0,
        "bp_diastolic": 90.0,
        "sex_F": 0.0,
        "sex_M": 1.0,
        "text": "Patient with hypertension"
    }
    
    response = requests.post(f"{BASE_URL}/predict", data=data)
    
    assert response.status_code == 200
    result = response.json()
    
    # Check required keys
    assert "patient_id" in result
    assert "risk_score" in result
    assert "triage_rank" in result
    assert "uncertainty" in result
    assert "risk_category" in result
    
    # Check types and ranges
    assert isinstance(result["risk_score"], float)
    assert 0.0 <= result["risk_score"] <= 1.0
    assert isinstance(result["uncertainty"], float)
    assert result["risk_category"] in ["Low", "Medium", "High"]
    
    print(f"✓ Prediction test passed")
    print(f"  - Risk score: {result['risk_score']}")
    print(f"  - Category: {result['risk_category']}")
    print(f"  - Uncertainty: {result['uncertainty']}")


def test_predict_with_image(wait_for_server):
    """Test prediction with image."""
    # Sample patient data
    data = {
        "patient_id": "TEST002",
        "age": 45.0,
        "bmi": 25.0,
        "chol": 180.0,
        "glucose": 95.0,
        "bp_systolic": 120.0,
        "bp_diastolic": 80.0,
        "sex_F": 1.0,
        "sex_M": 0.0
    }
    
    # Sample image
    image_path = Path("data/sample_radiology_image.png")
    
    if not image_path.exists():
        pytest.skip("Sample image not found")
    
    with open(image_path, 'rb') as f:
        files = {'image': ('xray.png', f, 'image/png')}
        response = requests.post(f"{BASE_URL}/predict", data=data, files=files)
    
    assert response.status_code == 200
    result = response.json()
    
    assert "risk_score" in result
    assert "uncertainty" in result
    
    print("✓ Prediction with image test passed")


def test_explain_shap(wait_for_server):
    """Test SHAP explanation endpoint."""
    data = {
        "patient_id": "TEST003",
        "age": 55.0,
        "bmi": 30.0,
        "chol": 220.0,
        "glucose": 110.0,
        "bp_systolic": 135.0,
        "bp_diastolic": 85.0,
        "sex_F": 0.0,
        "sex_M": 1.0,
        "explain_type": "shap"
    }
    
    response = requests.post(f"{BASE_URL}/explain", data=data)
    
    assert response.status_code == 200
    result = response.json()
    
    # Check required keys
    assert "patient_id" in result
    assert "explanation_type" in result
    assert result["explanation_type"] == "shap"
    assert "feature_importance" in result
    
    # Check feature importance
    importance = result["feature_importance"]
    assert isinstance(importance, dict)
    assert len(importance) > 0
    
    # Check that all features are present
    expected_features = ['age', 'bmi', 'chol', 'glucose', 'bp_systolic', 'bp_diastolic', 'sex_F', 'sex_M']
    for feat in expected_features:
        assert feat in importance
    
    print("✓ SHAP explanation test passed")
    print(f"  Top features:")
    sorted_features = sorted(importance.items(), key=lambda x: -x[1])
    for feat, imp in sorted_features[:3]:
        print(f"    - {feat}: {imp:.4f}")


def test_explain_missing_params(wait_for_server):
    """Test explanation with missing parameters."""
    data = {
        "age": 50.0,
        "explain_type": "shap"
        # Missing other required fields
    }
    
    response = requests.post(f"{BASE_URL}/explain", data=data)
    
    # Should return 422 Unprocessable Entity for missing fields
    assert response.status_code == 422
    
    print("✓ Missing parameters validation test passed")


def test_invalid_explain_type(wait_for_server):
    """Test explanation with invalid type."""
    data = {
        "patient_id": "TEST004",
        "age": 50.0,
        "bmi": 27.0,
        "chol": 200.0,
        "glucose": 100.0,
        "bp_systolic": 130.0,
        "bp_diastolic": 82.0,
        "sex_F": 1.0,
        "sex_M": 0.0,
        "explain_type": "invalid_type"
    }
    
    response = requests.post(f"{BASE_URL}/explain", data=data)
    
    # Should return 400 Bad Request for invalid explain_type
    assert response.status_code == 400
    
    print("✓ Invalid explain type validation test passed")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
