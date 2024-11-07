import pdb
import pytest
from fastapi.testclient import TestClient
from main import app
from endpoints.auth import create_access_token

client = TestClient(app)

@pytest.fixture
def admin_token():
    return create_access_token(data={"sub": "admin", "role": "admin"})

def test_train_endpoint_authorized(admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
   
    response = client.post("/train", json={
        "data": "data/dataset_Cramer.parquet",
        "model_type": "RandomForest",
        "n_estimators": 10,
        "max_depth": 5, 
        "retrain": False
    }, headers=headers)

    assert response.status_code == 200
    assert "Model RandomForest training started successfully" in response.json()["message"]

def test_train_endpoint_unauthorized():
    response = client.post("/train", json={})
    assert response.status_code == 401