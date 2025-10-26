from fastapi import APIRouter, HTTPException, Query
from app.integrations.mongo_repo import latest_training_run, list_training_runs, get_training_run, count_training_runs
from app.models.pipeline import predict_one
from app.views.responses import InputData
router = APIRouter(prefix="/api")

@router.get("/metrics/latest")
def metrics_latest():
    doc = latest_training_run()
    if not doc:
        raise HTTPException(status_code=404, detail="No hay métricas registradas. Ejecuta scripts/train.py primero.")
    m = doc.get("metrics", {})
    return {
        "run_id": doc.get("run_id"),
        "ts": doc.get("ts"),
        "Modelo": doc.get("model_name"),
        "Version": doc.get("model_version"),
        "Accuracy": m.get("accuracy"),
        "Precision": m.get("precision"),
        "Recall": m.get("recall"),
        "F1-Score": m.get("f1"),
        "ROC_AUC": m.get("roc_auc"),
        "Matriz_de_Confusion": m.get("confusion_matrix"),
        "Curvas": doc.get("curves"),
        "Features": doc.get("features_used", []),
        "Params": doc.get("params", {}),
    }

@router.get("/metrics")
def metrics_list(limit: int = Query(10, ge=1, le=100), page: int = Query(1, ge=1)):
    total = count_training_runs()
    items = list_training_runs(limit=limit, page=page)
    # opcional: listar sólo metadatos en vez de curvas para la lista
    for it in items:
        if "curves" in it:
            it.pop("curves")  # la lista es más ligera; detalle por id
    return {
        "total": total,
        "page": page,
        "limit": limit,
        "pages": (total + limit - 1) // limit,
        "items": items,
    }

@router.get("/metrics/{run_id}")
def metrics_detail(run_id: str):
    doc = get_training_run(run_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"run_id {run_id} no encontrado.")
    return doc

@router.post("/predict")
def predict(payload: InputData):
    """
    Predice si el cliente aceptará ('yes'/'no') y la probabilidad de 'yes'.
    """
    try:
        result = predict_one(payload.model_dump())
        return {
            "Modelo": "DecisionTreeClassifier",
            **result
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción: {e}")    
