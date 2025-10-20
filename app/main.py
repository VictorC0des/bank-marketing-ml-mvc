from fastapi import FastAPI
from app.controllers.api import router as api_router

app = FastAPI(title="Bank Marketing DecisionTree API", version="1.0")
app.include_router(api_router)

@app.get("/health")
def health():
    return {"status": "ok"}
