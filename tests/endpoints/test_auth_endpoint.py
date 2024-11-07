import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_login():
    response = client.post("/token", data={"username": "admin", "password": "admin123"})
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_login_invalid_credentials():
    response = client.post("/token", data={"username": "admin", "password": "wrongpassword"})
    assert response.status_code == 401