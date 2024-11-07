from fastapi.testclient import TestClient
from main import app
from endpoints.auth import create_access_token

client = TestClient(app)

@pytest.fixture
def user_token():
    return create_access_token(data={"sub": "user", "role": "user"})

def test_predict_endpoint(user_token):
    headers = {"Authorization": f"Bearer {user_token}"}
    response = client.post("/predict", json={
        "model_uri": "models:/KNN_Accident_Model/1",
        "data_path": "data/sample_data_for_prediction.parquet"
    }, headers=headers)
    assert response.status_code == 200
    assert "predictions" in response.json()
