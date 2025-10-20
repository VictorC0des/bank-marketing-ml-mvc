from fastapi import APIRouter, HTTPException
from app.views.responses import InputData
from app.models.pipeline import predict_one
import os, json

router = APIRouter(prefix="/api", tags=["API"])
METRICS_PATH = os.getenv("METRICS_PATH", "artifacts/metrics.json")

@router.post("/predict")
def predict(data: InputData):
    return predict_one(data.dict())

@router.get("/metrics")
def metrics():
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(status_code=404, detail="Métricas no disponibles. Ejecuta train.py")
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        m = json.load(f)
    # Tu metrics.json trae estas claves desde train.py (accuracy, classification_report, confusion_matrix, best_params)
    # :contentReference[oaicite:3]{index=3}
    return {
        "Modelo": "DecisionTreeClassifier",
        "Accuracy": m.get("accuracy"),
        "Precision_Recall_F1 (texto)": m.get("classification_report"),
        "Matriz_de_Confusion": m.get("confusion_matrix"),
        "Mejores_Hiperparametros": m.get("best_params")
    }
