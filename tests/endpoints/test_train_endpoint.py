
# tests/endpoints/test_train_endpoint.py
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from main import app
from endpoints.auth import create_access_token

client = TestClient(app)

@pytest.fixture
def admin_token():
    return create_access_token(data={"sub": "admin", "role": "admin"})

@patch("endpoints.train.run_experiment")
def test_train_endpoint_authorized(mock_run_experiment, admin_token):
    # Mock the return value of run_experiment
    mock_run_experiment.return_value = {
        "accuracy": 0.85,
        "precision": 0.8,
        "recall": 0.75,
        "f1_score": 0.77
    }

    headers = {"Authorization": f"Bearer {admin_token}"}

    # Call the /train endpoint with mocked data
    response = client.post("/train", json={
        "data": "data/dataset_Cramer.parquet",
        "model_type": "RandomForest",
        "n_neighbors": None,  # Explicitly set to None
        "weights": None,      # Explicitly set to None
        "n_estimators": 10,
        "max_depth": 5,
        "retrain": False
    }, headers=headers)

    # Assertions
    assert response.status_code == 200
    assert "Model RandomForest training started successfully" in response.json()["message"]
    assert response.json()["output"]["accuracy"] == 0.85
    assert response.json()["output"]["precision"] == 0.8
    assert response.json()["output"]["recall"] == 0.75
    assert response.json()["output"]["f1_score"] == 0.77

    # Verify that run_experiment was called with the correct arguments
    mock_run_experiment.assert_called_once_with(
        data_path="data/dataset_Cramer.parquet",
        model_type="RandomForest",
        n_neighbors=None,
        weights=None,
        n_estimators=10,
        max_depth=5,
        retrain=False
    )

def test_train_endpoint_unauthorized():
    # Call the /train endpoint without authorization
    response = client.post("/train", json={})
    assert response.status_code == 401