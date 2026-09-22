from fastapi.testclient import TestClient
from app import app
client = TestClient(app)

def test_home():
    r = client.get("/")
    assert r.status_code == 200

def test_health():
    r = client.get("/health")
    assert r.status_code == 200

def test_predict():
    r = client.post("/predict", json={"features":[0.5]*8})
    assert r.status_code == 200
    assert "prediction" in r.json()
