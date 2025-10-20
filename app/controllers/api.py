from fastapi import APIRouter

router = APIRouter(prefix="", tags=["api"])

@router.get("/")
def root():
    return {"message": "API scaffold listo"}
