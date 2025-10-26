# app/integrations/mongo_repo.py
import os, uuid, datetime
from typing import Dict, Any, Optional, List
from pymongo import MongoClient, ASCENDING, DESCENDING
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
MONGO_DB = os.getenv("MONGO_DB", "bank_ml")
COL_TRAIN = os.getenv("MONGO_TRAINING_COL", "training_runs")

_client = MongoClient(MONGO_URI)
_db = _client[MONGO_DB]
_training = _db[COL_TRAIN]  # la colección se crea al primer insert

# índice para ordenar por fecha (recomendado)
_training.create_index([("ts", DESCENDING)], name="ts_desc", background=True)

def save_training_run(doc: Dict[str, Any]) -> str:
    """Inserta un documento de métricas. Crea BD/colección si no existen."""
    if "run_id" not in doc:
        doc["run_id"] = str(uuid.uuid4())
    if "ts" not in doc:
        doc["ts"] = datetime.datetime.utcnow()
    _training.insert_one(doc)
    return doc["run_id"]

def latest_training_run() -> Optional[Dict[str, Any]]:
    """Última corrida por timestamp (sin _id)."""
    return _training.find_one(sort=[("ts", -1)], projection={"_id": 0})

def get_training_run(run_id: str) -> Optional[Dict[str, Any]]:
    return _training.find_one({"run_id": run_id}, {"_id": 0})

def list_training_runs(limit: int = 10, page: int = 1) -> List[Dict[str, Any]]:
    """Lista paginada, ordenada por fecha desc."""
    skip = max(page - 1, 0) * max(limit, 1)
    cursor = _training.find({}, {"_id": 0}).sort("ts", DESCENDING).skip(skip).limit(int(limit))
    return list(cursor)

def count_training_runs() -> int:
    return _training.count_documents({})
