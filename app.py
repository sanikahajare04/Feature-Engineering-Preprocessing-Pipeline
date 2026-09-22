from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="ML API - Sanika Hajare", version="1.0")
model = joblib.load("SVM_champion.pkl")

class Req(BaseModel):
    features: list # must be 8 numbers

@app.get("/")
def home():
    return {"status":"ok", "author":"Sanika Mahadev Hajare", "model":"SVM_champion.pkl", "n_features":8}

@app.get("/health")
def health():
    return {"model_loaded": True, "n_features": 8}

@app.post("/predict")
def predict(r: Req):
    X = np.array(r.features).reshape(1,-1)
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X).max()) if hasattr(model,"predict_proba") else 0.9
    return {"prediction": pred, "probability": prob, "model": "SVM_champion.pkl"}
