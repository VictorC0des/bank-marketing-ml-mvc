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

# ===== CONFIGURACIÓN PARA MACHINE LEARNING (Decision Tree) =====
PIPELINE_PATH = os.getenv("PIPELINE_PATH", "artifacts/decision_tree_model.joblib")
MODEL_CACHE_PATH = os.getenv("MODEL_CACHE_PATH", "artifacts/model_cached.joblib")

# ===== CONFIGURACIÓN PARA DEEP LEARNING (Neural Network) =====
DEEPLEARNING_MODEL_PATH = os.getenv("DEEPLEARNING_MODEL_PATH", "artifacts/deep_learning_model.h5")
DEEPLEARNING_MODEL_CACHE_PATH = os.getenv("DEEPLEARNING_MODEL_CACHE_PATH", "artifacts/dl_model_cached.h5")
DEEPLEARNING_SCALER_PATH = os.getenv("DEEPLEARNING_SCALER_PATH", "artifacts/dl_scaler.joblib")
DEEPLEARNING_SCALER_CACHE_PATH = os.getenv("DEEPLEARNING_SCALER_CACHE_PATH", "artifacts/dl_scaler_cached.joblib")
DEEPLEARNING_ENCODER_PATH = os.getenv("DEEPLEARNING_ENCODER_PATH", "artifacts/dl_encoder.joblib")
DEEPLEARNING_ENCODER_CACHE_PATH = os.getenv("DEEPLEARNING_ENCODER_CACHE_PATH", "artifacts/dl_encoder_cached.joblib")

# ===== CACHÉS GLOBALES SEPARADOS POR TIPO =====
# Machine Learning
_model_cache_ml = None
_model_meta_ml = {"source": None, "path": None, "run_id": None}

# Deep Learning
_model_cache_dl = None
_scaler_cache_dl = None
_encoder_cache_dl = None
_model_meta_dl = {"source": None, "path": None, "run_id": None}


def _download_from_gridfs(cache_path: str, fs_id: str = None) -> bool:
    """Descarga un artefacto desde GridFS a cache_path. Devuelve True si tuvo éxito."""
    if not fs_id:
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


# ============================================================================
# MACHINE LEARNING (Decision Tree) - FUNCIONES
# ============================================================================

def load_model_ml():
    """Carga el modelo Decision Tree más reciente. Usa archivo local o descarga desde GridFS."""
    global _model_cache_ml, _model_meta_ml
    if _model_cache_ml is None:
        path = PIPELINE_PATH
        if os.path.exists(path):
            _model_cache_ml = joblib.load(path)
            _model_meta_ml.update({"source": "file", "path": path})
        else:
            # Intenta descargar desde GridFS - buscar el modelo Decision Tree más reciente
            from integrations.mongo_repo import latest_artifact_info_by_type
            info = latest_artifact_info_by_type(model_type="decision_tree")
            if info:
                # Buscar artifact_fs_id o artifact_fs_id_model
                fs_id = info.get("artifact_fs_id_model") or info.get("artifact_fs_id")
                if fs_id and _download_from_gridfs(MODEL_CACHE_PATH, fs_id):
                    _model_cache_ml = joblib.load(MODEL_CACHE_PATH)
                    _model_meta_ml.update({"source": "gridfs", "path": MODEL_CACHE_PATH, "run_id": info.get("run_id")})
                    return _model_cache_ml
            
            # Fallback: intenta desde alias local
            alias = info.get("artifact_alias_model") or info.get("artifact_alias") if info else None
            if alias and os.path.exists(alias):
                _model_cache_ml = joblib.load(alias)
                _model_meta_ml.update({"source": "alias", "path": alias, "run_id": info.get("run_id") if info else None})
                return _model_cache_ml
            
            raise HTTPException(status_code=500, detail="No se puede cargar el modelo ML (archivo/GridFS no disponibles).")
    return _model_cache_ml


def predict_one_ml(payload: dict) -> dict:
    """Predice usando Decision Tree (ML clásico)."""
    model = load_model_ml()
    X = pd.DataFrame([payload])
    
    # NO aplicar featurización - el modelo fue entrenado con columnas originales
    # Solo mantener las columnas que el modelo espera
    
    try:
        yhat = model.predict(X)[0]
        proba_yes = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else None
        
        return {
            "Prediction": "yes" if yhat == 1 else "no",
            "Probability_yes": proba_yes
        }
    except Exception as e:
        # Si falla, intentar con featurización
        try:
            from integrations.featurize import featurize_df
            X = featurize_df(X)
            numeric_cols = X.select_dtypes(include=['int64', 'int32', 'float64', 'float32', 'bool']).columns.tolist()
            X = X[numeric_cols]
            
            yhat = model.predict(X)[0]
            proba_yes = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else None
            
            return {
                "Prediction": "yes" if yhat == 1 else "no",
                "Probability_yes": proba_yes
            }
        except Exception as e2:
            raise HTTPException(status_code=400, detail=f"Error en predicción: {str(e2)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción del modelo: {str(e)}")


# ============================================================================
# DEEP LEARNING (Neural Network) - FUNCIONES
# ============================================================================

def load_model_dl():
    """Carga el modelo Deep Learning más reciente (Keras/TensorFlow) + Scaler + Preprocessor desde GridFS."""
    global _model_cache_dl, _scaler_cache_dl, _encoder_cache_dl, _model_meta_dl
    
    if _model_cache_dl is None:
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            raise HTTPException(status_code=500, detail="TensorFlow no está instalado. Instala: pip install tensorflow keras")
        
        # Obtener info del modelo Deep Learning más reciente
        from integrations.mongo_repo import latest_artifact_info_by_type
        info = latest_artifact_info_by_type(model_type="deep_learning")
        
        if not info:
            raise HTTPException(status_code=404, detail="No hay modelo Deep Learning entrenado en la BD.")
        
        # ===== Carga del Modelo DL =====
        model_path = DEEPLEARNING_MODEL_PATH
        if os.path.exists(model_path):
            # Usar archivo local si existe
            try:
                _model_cache_dl = keras.models.load_model(model_path)
                _model_meta_dl.update({"source": "file", "path": model_path, "run_id": info.get("run_id")})
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error cargando modelo DL local: {e}")
        else:
            # Intentar descargar desde GridFS
            fs_id_model = info.get("artifact_fs_id_model")
            if fs_id_model:
                try:
                    if _download_from_gridfs(DEEPLEARNING_MODEL_CACHE_PATH, fs_id_model):
                        # Intentar cargar como joblib primero (Keras/TF modelos frecuentemente se guardan así)
                        try:
                            _model_cache_dl = joblib.load(DEEPLEARNING_MODEL_CACHE_PATH)
                            _model_meta_dl.update({"source": "gridfs", "format": "joblib", "path": DEEPLEARNING_MODEL_CACHE_PATH, "run_id": info.get("run_id")})
                        except Exception:
                            # Si joblib falla, intentar como H5
                            try:
                                _model_cache_dl = keras.models.load_model(DEEPLEARNING_MODEL_CACHE_PATH)
                                _model_meta_dl.update({"source": "gridfs", "format": "h5", "path": DEEPLEARNING_MODEL_CACHE_PATH, "run_id": info.get("run_id")})
                            except Exception as e_h5:
                                raise HTTPException(status_code=500, detail=f"Error cargando modelo DL (probado joblib y h5): {e_h5}")
                    else:
                        raise HTTPException(status_code=404, detail="No se pudo descargar el modelo DL desde GridFS.")
                except HTTPException:
                    raise
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error descargando modelo DL: {e}")
            else:
                raise HTTPException(status_code=404, detail="No hay artifact_fs_id_model en la BD para Deep Learning.")
        
        # ===== Carga del Preprocessor (Scaler) =====
        scaler_path = DEEPLEARNING_SCALER_PATH
        if os.path.exists(scaler_path):
            _scaler_cache_dl = joblib.load(scaler_path)
        else:
            # Intentar descargar desde GridFS
            fs_id_preprocessor = info.get("artifact_fs_id_preprocessor")
            if fs_id_preprocessor:
                try:
                    if _download_from_gridfs(DEEPLEARNING_SCALER_CACHE_PATH, fs_id_preprocessor):
                        _scaler_cache_dl = joblib.load(DEEPLEARNING_SCALER_CACHE_PATH)
                    else:
                        raise HTTPException(status_code=404, detail="No se pudo descargar el preprocessor desde GridFS.")
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error descargando preprocessor: {e}")
            else:
                raise HTTPException(status_code=404, detail="No hay artifact_fs_id_preprocessor en la BD para Deep Learning.")
        
        # ===== Carga del Encoder (Categorías) - Opcional =====
        encoder_path = DEEPLEARNING_ENCODER_PATH
        if os.path.exists(encoder_path):
            try:
                _encoder_cache_dl = joblib.load(encoder_path)
            except Exception:
                _encoder_cache_dl = None
        else:
            # El encoder es opcional, puede no existir
            _encoder_cache_dl = None
    
    return _model_cache_dl, _scaler_cache_dl, _encoder_cache_dl


def predict_one_dl(payload: dict) -> dict:
    """Predice usando Deep Learning. Soporta modelos Keras/TF y scikit-learn MLPClassifier."""
    try:
        model, scaler, encoder = load_model_dl()
    except HTTPException as e:
        # Si el modelo DL no está disponible, retornar error informativo
        raise HTTPException(status_code=503, detail=f"Modelo Deep Learning no disponible: {e.detail}")
    
    X = pd.DataFrame([payload])
    
    # Aplicar featurización (NECESARIA para Deep Learning - espera 67 features)
    try:
        from integrations.featurize import featurize_df
        X = featurize_df(X)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en featurización: {str(e)}")
    
    # Aplicar el scaler (ColumnTransformer) que hace one-hot encoding
    try:
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            # Si no hay scaler, seleccionar solo columnas numéricas
            numeric_cols = X.select_dtypes(include=['int64', 'int32', 'float64', 'float32', 'bool']).columns
            X_scaled = X[numeric_cols].values
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error escalando features: {str(e)}")
    
    try:
        # Detectar tipo de modelo
        model_type_name = type(model).__name__
        
        if model_type_name == 'MLPClassifier':
            # scikit-learn MLPClassifier
            proba = model.predict_proba(X_scaled)[0]
            proba_yes = float(proba[1]) if len(proba) > 1 else float(proba[0])
        else:
            # Keras/TensorFlow model
            proba = model.predict(X_scaled, verbose=0)[0]
            proba_yes = float(proba[1]) if len(proba) > 1 else float(proba[0])
        
        yhat = 1 if proba_yes >= 0.5 else 0
        
        return {
            "Prediction": "yes" if yhat == 1 else "no",
            "Probability_yes": proba_yes
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción del modelo DL: {str(e)}")


# ============================================================================
# FUNCIONES PÚBLICAS - WRAPPER PRINCIPAL
# ============================================================================

def predict_one(payload: dict, model_type: str = "decision_tree") -> dict:
    """
    Predice usando el modelo especificado.
    
    Args:
        payload: diccionario con los datos del cliente
        model_type: "decision_tree" o "deep_learning"
    
    Returns:
        dict con predicción y probabilidad
    """
    if model_type == "decision_tree":
        return predict_one_ml(payload)
    elif model_type == "deep_learning":
        return predict_one_dl(payload)
    else:
        raise ValueError(f"Tipo de modelo no reconocido: {model_type}. Usa 'decision_tree' o 'deep_learning'.")


def refresh_model(run_id: str = None, model_type: str = "decision_tree") -> dict:
    """Fuerza la recarga del modelo desde GridFS."""
    global _model_cache_ml, _model_cache_dl, _scaler_cache_dl, _encoder_cache_dl, _model_meta_ml, _model_meta_dl
    
    if model_type == "decision_tree":
        _model_cache_ml = None
        _model_meta_ml = {"source": None, "path": None, "run_id": None}
        model = load_model_ml()
        return _model_meta_ml
    elif model_type == "deep_learning":
        _model_cache_dl = None
        _scaler_cache_dl = None
        _encoder_cache_dl = None
        _model_meta_dl = {"source": None, "path": None, "run_id": None}
        load_model_dl()
        return _model_meta_dl
    else:
        raise ValueError(f"Tipo de modelo no reconocido: {model_type}")


def model_info() -> dict:
    """Devuelve info básica de ambos modelos."""
    ml_info = {
        "decision_tree": {
            "source": _model_meta_ml.get("source"),
            "path": _model_meta_ml.get("path"),
            "run_id": _model_meta_ml.get("run_id"),
            "pipeline_path": PIPELINE_PATH,
            "pipeline_exists": os.path.exists(PIPELINE_PATH),
            "cache_path": MODEL_CACHE_PATH,
            "cache_exists": os.path.exists(MODEL_CACHE_PATH),
            "loaded": _model_cache_ml is not None,
        }
    }
    
    dl_info = {
        "deep_learning": {
            "model_path": DEEPLEARNING_MODEL_PATH,
            "model_exists": os.path.exists(DEEPLEARNING_MODEL_PATH),
            "scaler_path": DEEPLEARNING_SCALER_PATH,
            "scaler_exists": os.path.exists(DEEPLEARNING_SCALER_PATH),
            "encoder_path": DEEPLEARNING_ENCODER_PATH,
            "encoder_exists": os.path.exists(DEEPLEARNING_ENCODER_PATH),
            "model_loaded": _model_cache_dl is not None,
            "scaler_loaded": _scaler_cache_dl is not None,
            "encoder_loaded": _encoder_cache_dl is not None,
        }
    }
    
    return {**ml_info, **dl_info}

