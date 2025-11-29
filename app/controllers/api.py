from fastapi import APIRouter, HTTPException, Query
from integrations.mongo_repo import (
    latest_training_run, list_training_runs, get_training_run, count_training_runs,
    latest_training_run_by_type, list_training_runs_by_type
)
from app.models.pipeline import predict_one, refresh_model, model_info
import math
import numbers
from datetime import date, datetime
try:
    from bson import ObjectId
except Exception:  # pragma: no cover
    ObjectId = None  # type: ignore


def _sanitize_for_json(obj):
    """Convierte NaN/Inf a None y numpy arrays a listas, recursivamente."""
    # numpy arrays/Series con tolist
    if hasattr(obj, "tolist"):
        return _sanitize_for_json(obj.tolist())
    # ObjectId
    if ObjectId is not None and isinstance(obj, ObjectId):
        return str(obj)
    # datetime/date
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    # dict
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    # list/tuple
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(x) for x in obj]
    # números (incluye numpy.number)
    if isinstance(obj, numbers.Number):
        try:
            x = float(obj)
            if not math.isfinite(x):
                return None
            return x if isinstance(obj, float) else obj
        except Exception:
            return None
    return obj
from app.views.responses import InputData
router = APIRouter(prefix="/api")

@router.get("/metrics/latest")
def metrics_latest(
    model_type: str = Query(None, enum=["decision_tree", "deep_learning"], description="Filtrar por tipo de modelo (opcional)"),
    include_curves: bool = Query(True, description="Incluir curvas ROC y PR.")
):
    if model_type:
        doc = latest_training_run_by_type(model_type=model_type, include_curves=include_curves)
    else:
        doc = latest_training_run(include_curves=include_curves)
    
    if not doc:
        raise HTTPException(status_code=404, detail="No hay métricas registradas. Ejecuta scripts/train.py primero.")
    # Devuelve el documento tal cual se guarda en Mongo (con _id), sanitizado para JSON
    return _sanitize_for_json(doc if include_curves else {**doc, "curves": None})

@router.get("/metrics")
def metrics_list(
    model_type: str = Query(None, enum=["decision_tree", "deep_learning"], description="Filtrar por tipo de modelo (opcional)"),
    limit: int = Query(10, description="Resultados por página. Usa -1 para traer todos."),
    page: int = Query(1, ge=1, description="Número de página."),
    include_curves: bool = Query(False, description="Incluir curvas ROC/PR en cada registro.")
):
    if model_type:
        total = len(list_training_runs_by_type(model_type=model_type, limit=-1, include_curves=False))
        items = list_training_runs_by_type(model_type=model_type, limit=limit, page=page, include_curves=include_curves)
    else:
        total = count_training_runs()
        items = list_training_runs(limit=limit, page=page, include_curves=include_curves)

    # Si no queremos curvas en la lista, las quitamos para que sea más ligera
    if not include_curves:
        for it in items:
            it.pop("curves", None)

    for it in items:
        it.pop("features_used", None)

    response_payload = {
        "total": total,
        "page": page,
        "limit": limit,
        "pages": 1 if limit <= 0 else (total + limit - 1) // limit,
        "items": items,
    }

    return _sanitize_for_json(response_payload)

@router.get("/metrics/{run_id}")
def metrics_detail(run_id: str, include_curves: bool = Query(True, description="Incluir curvas ROC/PR del modelo específico.")):
    doc = get_training_run(run_id, include_curves=include_curves)
    if not doc:
        raise HTTPException(status_code=404, detail=f"run_id {run_id} no encontrado.")
    return _sanitize_for_json(doc)

@router.post("/predict")
def predict(
    payload: InputData,
    model_type: str = Query(
        "decision_tree",
        enum=["decision_tree", "deep_learning"],
        description="Tipo de modelo a usar: 'decision_tree' o 'deep_learning'"
    )
):
    """
    Predice si el cliente aceptará ('yes'/'no') y la probabilidad de 'yes'.
    
    Parámetros:
    - payload: InputData con datos del cliente
    - model_type: Tipo de modelo ('decision_tree' o 'deep_learning')
    """
    try:
        result = predict_one(payload.model_dump(), model_type=model_type)
        
        # Nombre del modelo para la respuesta
        model_name = "DecisionTreeClassifier" if model_type == "decision_tree" else "DeepLearningNN"
        
        return {
            "Modelo": model_name,
            "model_type": model_type,
            **result
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción: {e}")


@router.get("/model/refresh")
def model_refresh(
    run_id: str | None = Query(default=None, description="Opcional: forzar cargar artefacto de un run_id específico."),
    model_type: str = Query(default="decision_tree", enum=["decision_tree", "deep_learning"], description="Tipo de modelo a refrescar")
):
    """Descarga/carga el modelo más reciente desde GridFS/alias y devuelve el origen."""
    meta = refresh_model(run_id, model_type=model_type)
    return {"status": "ok", "model_type": model_type, **meta}


@router.get("/model/health")
def model_health():
    """Devuelve estado de ambos modelos (ML y DL) cargados y rutas disponibles."""
    info = model_info()
    return {
        "status": "ok",
        "models": info
    }
