"""Repo Mongo (API): solo lectura de runs de entrenamiento."""
import os
from typing import Dict, Any, Optional, List
from pymongo import MongoClient, DESCENDING
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")
COL_TRAIN = os.getenv("MONGO_TRAINING_COL")

_client = MongoClient(MONGO_URI)
_db = _client[MONGO_DB]
_training = _db[COL_TRAIN]
_training.create_index([("ts", DESCENDING)], name="ts_desc", background=True)

def latest_training_run(include_curves: bool = True) -> Optional[Dict[str, Any]]:
    # Incluir _id por compatibilidad con consumidores
    proj = None if include_curves else {"curves": 0}
    return _training.find_one(sort=[("ts", -1)], projection=proj)

def get_training_run(run_id: str, include_curves: bool = True) -> Optional[Dict[str, Any]]:
    proj = None if include_curves else {"curves": 0}
    return _training.find_one({"run_id": run_id}, proj)

def count_training_runs() -> int:
    return _training.count_documents({})

def list_training_runs(limit: int = 10, page: int = 1, include_curves: bool = False) -> List[Dict[str, Any]]:
    """
    - limit > 0 => paginado
    - limit <= 0 => devuelve TODOS
    """
    proj = None if include_curves else {"curves": 0}

    if limit and limit > 0:
        skip = max(page - 1, 0) * limit
        cursor = _training.find({}, proj).sort("ts", -1).skip(skip).limit(int(limit))
        return list(cursor)
    else:
        # sin paginar: todos
        cursor = _training.find({}, proj).sort("ts", -1)
        return list(cursor)


def artifact_info_by_run(run_id: str) -> Optional[Dict[str, Any]]:
    """Devuelve metadatos de artefacto por run_id (solo lectura)."""
    proj = {
        "_id": 0,
        "run_id": 1,
        "artifact_fs_id": 1,
        "artifact_alias": 1,
        "artifact_path": 1,
        "artifact_size_bytes": 1,
        "ts": 1,
    }
    return _training.find_one({"run_id": run_id}, projection=proj)


def latest_artifact_info() -> Optional[Dict[str, Any]]:
    """Devuelve metadatos de artefacto del run más reciente (solo lectura)."""
    proj = {
        "_id": 0,
        "run_id": 1,
        "artifact_fs_id": 1,
        "artifact_alias": 1,
        "artifact_path": 1,
        "artifact_size_bytes": 1,
        "ts": 1,
    }
    return _training.find_one(sort=[("ts", -1)], projection=proj)


def latest_training_run_by_type(model_type: str = "decision_tree", include_curves: bool = True) -> Optional[Dict[str, Any]]:
    """
    Devuelve el run más reciente de un tipo de modelo específico.
    
    Maneja tanto documentos nuevos (con model_type) como antiguos (sin model_type).
    Infiere el tipo por el nombre del modelo si no existe el campo model_type.
    
    Args:
        model_type: "decision_tree" o "deep_learning"
        include_curves: incluir curvas ROC/PR
    """
    proj = None if include_curves else {"curves": 0}
    
    # Buscar primero documentos con model_type explícito
    query = {"model_type": model_type}
    result = _training.find_one(query, sort=[("ts", -1)], projection=proj)
    
    if result:
        return result
    
    # Si no hay documentos con model_type, buscar por nombre de modelo
    # DecisionTree* -> decision_tree, DeepLearningNN* -> deep_learning
    if model_type == "decision_tree":
        model_name_pattern = "DecisionTree"
    elif model_type == "deep_learning":
        model_name_pattern = "DeepLearningNN"
    else:
        return None
    
    query = {"model_name": {"$regex": f"^{model_name_pattern}"}}
    return _training.find_one(query, sort=[("ts", -1)], projection=proj)


def list_training_runs_by_type(model_type: str = None, limit: int = 10, page: int = 1, include_curves: bool = False) -> List[Dict[str, Any]]:
    """
    Lista runs de entrenamiento filtrados opcionalmente por tipo de modelo.
    
    Maneja tanto documentos nuevos (con model_type) como antiguos (sin model_type).
    Infiere el tipo por el nombre del modelo si no existe el campo model_type.
    
    Args:
        model_type: "decision_tree", "deep_learning", o None para todos
        limit: registros por página (<=0 para todos)
        page: número de página
        include_curves: incluir curvas ROC/PR
    """
    proj = None if include_curves else {"curves": 0}
    
    # Construir query
    if model_type == "decision_tree":
        # Buscar por model_type o por nombre (para docs antiguos)
        query = {
            "$or": [
                {"model_type": "decision_tree"},
                {"model_name": {"$regex": "^DecisionTree"}}
            ]
        }
    elif model_type == "deep_learning":
        query = {
            "$or": [
                {"model_type": "deep_learning"},
                {"model_name": {"$regex": "^DeepLearningNN"}}
            ]
        }
    else:
        query = {}
    
    if limit and limit > 0:
        skip = max(page - 1, 0) * limit
        cursor = _training.find(query, proj).sort("ts", -1).skip(skip).limit(int(limit))
        return list(cursor)
    else:
        # sin paginar: todos
        cursor = _training.find(query, proj).sort("ts", -1)
        return list(cursor)


def latest_artifact_info_by_type(model_type: str = "decision_tree") -> Optional[Dict[str, Any]]:
    """
    Devuelve metadatos de artefacto del run más reciente de un modelo específico.
    
    Maneja tanto documentos nuevos (con model_type) como antiguos (sin model_type).
    """
    proj = {
        "_id": 0,
        "run_id": 1,
        "model_type": 1,
        "model_name": 1,
        "artifact_fs_id": 1,
        "artifact_fs_id_model": 1,
        "artifact_fs_id_preprocessor": 1,
        "artifact_alias": 1,
        "artifact_alias_model": 1,
        "artifact_path": 1,
        "artifact_size_bytes": 1,
        "ts": 1,
    }
    
    # Construir query
    if model_type == "decision_tree":
        query = {
            "$or": [
                {"model_type": "decision_tree"},
                {"model_name": {"$regex": "^DecisionTree"}}
            ]
        }
    elif model_type == "deep_learning":
        query = {
            "$or": [
                {"model_type": "deep_learning"},
                {"model_name": {"$regex": "^DeepLearningNN"}}
            ]
        }
    else:
        query = {}
    
    return _training.find_one(query, sort=[("ts", -1)], projection=proj)
