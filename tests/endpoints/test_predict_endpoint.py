import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from main import app
from endpoints.auth import create_access_token

client = TestClient(app)

@pytest.fixture
def user_token():
    return create_access_token(data={"sub": "user", "role": "user"})

@patch("mlflow.pyfunc.load_model")
def test_predict_endpoint(mock_load_model, user_token):
    # Mock the model's predict method
    mock_model = mock_load_model.return_value
    mock_model.predict.return_value = [0,1,1]

    headers = {"Authorization": f"Bearer {user_token}"}
    response = client.post("/predict", json={
        "model_uri": "models:/KNN_Accident_Model/1",
        "data_path": "data/sample_data_for_prediction.parquet"
    }, headers=headers)

    assert response.status_code == 200
    assert "predictions" in response.json()
    assert response.json()["predictions"] == ["Minor Accident", "Severe Accident", "Severe Accident"]