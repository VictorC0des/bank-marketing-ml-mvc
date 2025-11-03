import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.controllers.api import router as api_router

# Deshabilita documentación pública (Swagger/Redoc/OpenAPI) permanentemente
app = FastAPI(
    title="Bank Marketing DecisionTree API",
    version="1.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# Configuración CORS por entorno
_allow_origins_env = os.getenv("ALLOW_ORIGINS", "*")
if _allow_origins_env.strip() == "*" or _allow_origins_env.strip() == "":
    allow_origins = ["*"]
else:
    allow_origins = [o.strip() for o in _allow_origins_env.split(",") if o.strip()]

# Si se usan credenciales, es recomendable especificar orígenes concretos
_allow_credentials_env = os.getenv("ALLOW_CREDENTIALS", "false").lower() == "true"
allow_credentials = _allow_credentials_env and allow_origins != ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

@app.get("/health")
def health():
    return {"status": "ok"}
