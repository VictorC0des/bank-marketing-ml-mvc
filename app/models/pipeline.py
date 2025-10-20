import os
import joblib
import pandas as pd
from fastapi import HTTPException

PIPELINE_PATH = os.getenv("PIPELINE_PATH", "artifacts/decision_tree_model.joblib")
_model_cache = None

def load_model():
    global _model_cache
    if _model_cache is None:
        if not os.path.exists(PIPELINE_PATH):
            raise HTTPException(status_code=500, detail="Modelo no encontrado. Entrena primero.")
        _model_cache = joblib.load(PIPELINE_PATH)
    return _model_cache

def predict_one(payload: dict):
    model = load_model()
    X = pd.DataFrame([payload])
    yhat = model.predict(X)[0]
    proba_yes = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else None
    return {"Prediction": "yes" if yhat == 1 else "no", "Probability_yes": proba_yes}
