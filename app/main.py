from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.controllers.api import router as api_router

app = FastAPI(title="Bank Marketing DecisionTree API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], #Solo temporal, es para evitar que hay conflictos al consumir el api desde el front
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)   

app.include_router(api_router)

@app.get("/health")
def health():
    return {"status": "ok"}
