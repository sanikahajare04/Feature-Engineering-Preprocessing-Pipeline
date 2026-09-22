from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="ML API - Sanika", version="1.0")
model = joblib.load("SVM_champion.pkl")

# Auto-get column names from pipeline
try:
    COLS = list(model.feature_names_in_)
except:
    try:
        COLS = list(model.steps[0][1].feature_names_in_)
    except:
        COLS = [f"feature_{i}" for i in range(8)]

print(f"Columns: {COLS}")

class Req(BaseModel):
    features: dict # {"col_name": value}

@app.get("/")
def home():
    return {"status":"ok", "columns": COLS, "author":"Sanika"}

@app.get("/health")
def health():
    return {"model_loaded": True, "columns": COLS, "n_features": len(COLS)}

@app.post("/predict")
def predict(r: Req):
    # Convert dict to DataFrame for ColumnTransformer
    df = pd.DataFrame([r.features])
    # Ensure all columns present
    for c in COLS:
        if c not in df.columns:
            df[c] = 0
    df = df[COLS]
    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df).max()) if hasattr(model,"predict_proba") else 0.9
    return {"prediction": pred, "probability": prob, "model": "SVM_champion.pkl"}
