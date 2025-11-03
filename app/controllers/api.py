from fastapi import APIRouter, HTTPException, Query
from integrations.mongo_repo import latest_training_run, list_training_runs, get_training_run, count_training_runs
from app.models.pipeline import predict_one, refresh_model, model_info
from app.views.responses import InputData
router = APIRouter(prefix="/api")

@router.get("/metrics/latest")
def metrics_latest(include_curves: bool = Query(True, description="Incluir curvas ROC y PR.")):
    doc = latest_training_run(include_curves=include_curves)
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
        "Average_Precision": m.get("average_precision"),
        "Matriz_de_Confusion": m.get("confusion_matrix"),
        "Curvas": doc.get("curves"),
        "Params": doc.get("params", {}),
    }

@router.get("/metrics")
def metrics_list(
    limit: int = Query(10, description="Resultados por página. Usa -1 para traer todos."),
    page: int = Query(1, ge=1, description="Número de página."),
    include_curves: bool = Query(False, description="Incluir curvas ROC/PR en cada registro.")
):
    total = count_training_runs()
    items = list_training_runs(limit=limit, page=page, include_curves=include_curves)

    # Si no queremos curvas en la lista, las quitamos para que sea más ligera
    if not include_curves:
        for it in items:
            it.pop("curves", None)

    for it in items:
        it.pop("features_used", None)

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "pages": 1 if limit <= 0 else (total + limit - 1) // limit,
        "items": items,
    }

@router.get("/metrics/{run_id}")
def metrics_detail(run_id: str, include_curves: bool = Query(True, description="Incluir curvas ROC/PR del modelo específico.")):
    doc = get_training_run(run_id, include_curves=include_curves)
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
        return {"Modelo": "DecisionTreeClassifier", **result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción: {e}")


@router.get("/model/refresh")
def model_refresh(run_id: str | None = Query(default=None, description="Opcional: forzar cargar artefacto de un run_id específico.")):
    """Descarga/carga el modelo más reciente (o por run_id) desde GridFS/alias y devuelve el origen."""
    meta = refresh_model(run_id)
    return {"status": "ok", **meta}


@router.get("/model/health")
def model_health():
    """Devuelve estado del modelo cargado y rutas disponibles."""
    return model_info()
