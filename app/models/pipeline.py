import os
import sys
import shutil
import joblib
import pandas as pd
from fastapi import HTTPException
from pymongo import MongoClient
import gridfs
from bson import ObjectId
from dotenv import load_dotenv
from integrations.mongo_repo import latest_artifact_info, artifact_info_by_run

load_dotenv()

# Asegura que la raíz del proyecto esté en sys.path y registra alias del featurizer
try:
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
    # Importa la versión canónica y permite que registre el alias 'integrations.featurize'
    import integrations.featurize  # noqa: F401
except Exception:
    # Intento secundario: si falla, intenta el path antiguo
    try:
        __import__("integrations.featurize")
    except Exception:
        pass

PIPELINE_PATH = os.getenv("PIPELINE_PATH", "artifacts/decision_tree_model.joblib")
MODEL_CACHE_PATH = os.getenv("MODEL_CACHE_PATH", "artifacts/model_cached.joblib")
_model_cache = None
_model_meta = {"source": None, "path": None, "run_id": None}

def _download_from_gridfs(cache_path: str) -> bool:
    """Descarga el último modelo desde GridFS a cache_path. Devuelve True si tuvo éxito."""
    info = latest_artifact_info()
    if not info:
        return False
    fs_id = info.get("artifact_fs_id")
    # Si no hay fs_id pero existe artifact_alias local, úsalo como fallback local
    if not fs_id:
        alias = info.get("artifact_alias")
        if alias and os.path.exists(alias):
            try:
                mdl = joblib.load(alias)
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                joblib.dump(mdl, cache_path)
                return True
            except Exception:
                return False
        return False

    mongo_uri = os.getenv("MONGO_URI")
    mongo_db = os.getenv("MONGO_DB")
    if not mongo_uri or not mongo_db:
        return False
    try:
        # fs_id puede venir como str; convertir a ObjectId si aplica
        if isinstance(fs_id, str):
            try:
                fs_id = ObjectId(fs_id)
            except Exception:
                pass
        client = MongoClient(mongo_uri)
        db = client[mongo_db]
        fs = gridfs.GridFS(db)
        grid_out = fs.get(fs_id)
        data = grid_out.read()
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            f.write(data)
        return True
    except Exception:
        return False


def refresh_model(run_id: str | None = None) -> dict:
    """Fuerza la recarga del modelo: baja el más reciente (o por run_id) desde GridFS o usa alias local."""
    global _model_cache, _model_meta
    info = artifact_info_by_run(run_id) if run_id else latest_artifact_info()
    if not info:
        raise HTTPException(status_code=404, detail="No hay runs de entrenamiento con artefactos.")

    fs_id = info.get("artifact_fs_id")
    alias = info.get("artifact_alias")

    # Intenta GridFS primero
    mongo_uri = os.getenv("MONGO_URI")
    mongo_db = os.getenv("MONGO_DB")
    if fs_id and mongo_uri and mongo_db:
        try:
            # Convertir a ObjectId si viene como str
            if isinstance(fs_id, str):
                try:
                    fs_id = ObjectId(fs_id)
                except Exception:
                    pass
            client = MongoClient(mongo_uri)
            db = client[mongo_db]
            fs = gridfs.GridFS(db)
            grid_out = fs.get(fs_id)
            data = grid_out.read()
            os.makedirs(os.path.dirname(MODEL_CACHE_PATH), exist_ok=True)
            with open(MODEL_CACHE_PATH, "wb") as f:
                f.write(data)
            _model_cache = joblib.load(MODEL_CACHE_PATH)
            _model_meta = {"source": "gridfs", "path": MODEL_CACHE_PATH, "run_id": info.get("run_id")}
            return _model_meta
        except Exception:
            pass

    # Fallback: alias local si existe
    if alias and os.path.exists(alias):
        try:
            os.makedirs(os.path.dirname(MODEL_CACHE_PATH), exist_ok=True)
            shutil.copyfile(alias, MODEL_CACHE_PATH)
        except Exception:
            # Si falla la copia, igual intentamos cargar desde alias directamente
            pass
        try:
            _model_cache = joblib.load(MODEL_CACHE_PATH if os.path.exists(MODEL_CACHE_PATH) else alias)
            _model_meta = {"source": "alias", "path": (MODEL_CACHE_PATH if os.path.exists(MODEL_CACHE_PATH) else alias), "run_id": info.get("run_id")}
            return _model_meta
        except Exception:
            pass

    raise HTTPException(status_code=500, detail="No se pudo cargar el modelo (GridFS/alias no disponibles).")

def load_model():
    global _model_cache, _model_meta
    if _model_cache is None:
        path = PIPELINE_PATH
        if os.path.exists(path):
            _model_cache = joblib.load(path)
            _model_meta.update({"source": "file", "path": path})
        else:
            ok = _download_from_gridfs(MODEL_CACHE_PATH)
            if not ok or not os.path.exists(MODEL_CACHE_PATH):
                # Como último recurso, intenta refresh_model() que contempla alias
                meta = refresh_model()
                _model_meta.update(meta)
                return _model_cache
            _model_cache = joblib.load(MODEL_CACHE_PATH)
            _model_meta.update({"source": "gridfs", "path": MODEL_CACHE_PATH})
    return _model_cache


def model_info() -> dict:
    """Devuelve info básica del modelo cargado/cargable."""
    exists_file = os.path.exists(PIPELINE_PATH)
    exists_cache = os.path.exists(MODEL_CACHE_PATH)
    return {
        "source": _model_meta.get("source"),
        "path": _model_meta.get("path"),
        "run_id": _model_meta.get("run_id"),
        "pipeline_path": PIPELINE_PATH,
        "pipeline_exists": exists_file,
        "cache_path": MODEL_CACHE_PATH,
        "cache_exists": exists_cache,
        "loaded": _model_cache is not None,
    }

def predict_one(payload: dict):
    model = load_model()
    X = pd.DataFrame([payload])
    yhat = model.predict(X)[0]
    proba_yes = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else None
    return {"Prediction": "yes" if yhat == 1 else "no", "Probability_yes": proba_yes}
