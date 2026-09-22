from fastapi.testclient import TestClient
from app import app
import joblib
client = TestClient(app)

try:
    m = joblib.load("SVM_champion.pkl")
    COLS = list(m.feature_names_in_)
except:
    COLS = list(joblib.load("SVM_champion.pkl").steps[0][1].feature_names_in_)

def test_home():
    assert client.get("/").status_code == 200

def test_health():
    assert client.get("/health").status_code == 200

def test_predict():
    # Build dummy dict with correct column names
    sample = {c: 0.5 for c in COLS}
    r = client.post("/predict", json={"features": sample})
    print(r.text)
    assert r.status_code == 200
