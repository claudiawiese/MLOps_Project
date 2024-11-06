import pytest
from fastapi.testclient import TestClient
from jose import jwt
from datetime import timedelta
from main import app, SECRET_KEY, ALGORITHM, create_access_token

client = TestClient(app)

# Helper function to create token
def get_token_for_user(username: str):
    access_token = create_access_token(data={"sub": username, "role": "user"}, expires_delta=timedelta(minutes=30))
    return access_token

# Test the /token endpoint for successful and failed login
@pytest.mark.asyncio
async def test_login_success():
    response = client.post("/token", data={"username": "admin", "password": "admin123"})
    assert response.status_code == 200
    assert "access_token" in response.json()

@pytest.mark.asyncio
async def test_login_failure():
    response = client.post("/token", data={"username": "admin", "password": "wrongpassword"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Nom d'utilisateur ou mot de passe incorrect"