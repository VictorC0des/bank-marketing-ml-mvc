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
    proj = {"_id": 0}
    if not include_curves:
        proj["curves"] = 0
    return _training.find_one(sort=[("ts", -1)], projection=proj)

def get_training_run(run_id: str, include_curves: bool = True) -> Optional[Dict[str, Any]]:
    proj = {"_id": 0}
    if not include_curves:
        proj["curves"] = 0
    return _training.find_one({"run_id": run_id}, proj)

def count_training_runs() -> int:
    return _training.count_documents({})

def list_training_runs(limit: int = 10, page: int = 1, include_curves: bool = False) -> List[Dict[str, Any]]:
    """
    - limit > 0 => paginado
    - limit <= 0 => devuelve TODOS
    """
    proj = {"_id": 0}
    if not include_curves:
        proj["curves"] = 0

    if limit and limit > 0:
        skip = max(page - 1, 0) * limit
        cursor = _training.find({}, proj).sort("ts", -1).skip(skip).limit(int(limit))
        return list(cursor)
    else:
        # sin paginar: todos
        cursor = _training.find({}, proj).sort("ts", -1)
        return list(cursor)
