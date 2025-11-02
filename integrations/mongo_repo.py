import os
import uuid
import datetime
from typing import Dict, Any, Optional
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
MONGO_DB = os.getenv("MONGO_DB", "bank_ml")
COL_TRAIN = os.getenv("MONGO_TRAINING_COL", "training_runs")

_client: Optional[MongoClient] = None
_coll = None

def _get_collection():
    global _client, _coll
    if _client is None:
        _client = MongoClient(MONGO_URI)
    if _coll is None:
        db = _client[MONGO_DB]
        _coll = db[COL_TRAIN]
    return _coll


def save_training_run(doc: Dict[str, Any]) -> str:
    if not isinstance(doc, dict):
        raise TypeError("doc debe ser un dict")
    coll = _get_collection()
    if "run_id" not in doc or not doc.get("run_id"):
        doc["run_id"] = str(uuid.uuid4())
    if "ts" not in doc or not doc.get("ts"):
        doc["ts"] = datetime.datetime.utcnow()
    coll.insert_one(doc)
    return doc["run_id"]
