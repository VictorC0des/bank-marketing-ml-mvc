# Bank Marketing - DecisionTree API (Scaffold)

Estructura base (MVC) para el proyecto.  
Por ahora solo incluye la API mínima y directorios para datos, artefactos, scripts y pruebas.

## Ejecutar local
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
uvicorn app.main:app --reload
```

- Health: http://127.0.0.1:8000/health
- Docs: http://127.0.0.1:8000/docs

## Estructura
- `app/` API y módulos
- `scripts/` entrenamientos
- `data/` datasets (no se suben)
- `artifacts/` modelos/metrics (no se suben)
- `tests/` pruebas

## Próximos pasos
- Añadir preprocesamiento (One-Hot/Ordinal) y árbol de decisión
- Script de entrenamiento real en `scripts/train.py`
- Endpoints `/predict` y `/metrics`
- CI/CD a Cloud Run
