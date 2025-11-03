# 📊 Bank Marketing ML API

API de predicción basada en **FastAPI** para estimar si un cliente aceptará una oferta de depósito a plazo, usando un modelo de Machine Learning (árbol de decisión) entrenado con el dataset de marketing bancario.

---

## 🚀 Descripción general

Este proyecto permite:

- Exponer endpoints REST para consultar métricas, estado del modelo y realizar predicciones.
- Realizar predicciones en línea mediante el modelo cargado.
- Leer y servir métricas almacenadas en **MongoDB** (si está configurado con variables `MONGO_*`).

Importante: la documentación interactiva (Swagger/Redoc) está deshabilitada en producción. Usa Postman o curl para consumir la API.

---

## 🧱 Estructura del proyecto

```
bank-marketing-ml-mvc/
│
├── app/
│   ├── controllers/
│   │   └── api.py            # Endpoints de la API (prefijo /api)
│   ├── integrations/
│   │   └── mongo_repo.py     # Acceso a MongoDB/GridFS
│   ├── models/
│   │   └── pipeline.py       # Carga/refresh del modelo y predicción
│   └── main.py               # FastAPI app, CORS y /health
│
├── artifacts/
│   └── decision_tree_model.joblib  # Modelo por defecto (si no se usa GridFS)
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuración (variables de entorno)

Estas variables son leídas por la API:

- PIPELINE_PATH: ruta local del modelo por defecto. Ej: `artifacts/decision_tree_model.joblib`.
- MODEL_CACHE_PATH: ruta donde se cachea el modelo descargado de GridFS. Ej: `artifacts/model_cached.joblib`.
- MONGO_URI: conexión a MongoDB (requerido si se usan métricas/artifacts desde BD).
- MONGO_DB: base de datos MongoDB. Ej: `bank_marketing`.
- MONGO_TRAINING_COL: colección donde se guardan métricas de entrenamiento. Por defecto: `training_runs`.
- ALLOW_ORIGINS: orígenes permitidos para CORS (coma-separados). Ej: `http://localhost:3000,http://app.local`.
- ALLOW_CREDENTIALS: `true|false`. Si `true`, no usar `*` en ALLOW_ORIGINS.

---

## 🌐 Ejecutar la API

Local (hot-reload):

```bash
uvicorn app.main:app --reload
```

Docker (respeta $PORT):

```bash
docker build -t bank-api:local .
docker run --rm -p 8000:8000 --env-file .env bank-api:local
```

Una vez levantada, la base URL es `http://localhost:8000`.

---

## � Referencia de la API (para frontend)

Prefijo común: `/api` (excepto `/health`).

### 1) GET `/health`
Ping del servicio.

- Respuesta 200
  ```json
  { "status": "ok" }
  ```

---

### 2) POST `/api/predict`
Predice si el cliente aceptará la oferta.

- Body (JSON) — esquema `InputData`:
  - age: int
  - job: uno de ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"]
  - marital: ["married", "single", "divorced", "unknown"]
  - education: ["unknown", "primary", "secondary", "tertiary"]
  - default, housing, loan: ["yes", "no", "unknown"]
  - contact: ["cellular", "telephone", "unknown"]
  - day: 1–31
  - month: ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
  - duration: >= 0
  - campaign: >= 1
  - pdays: int (usar -1 si nunca contactado)
  - previous: >= 0

- Ejemplo de request (JSON):
  ```json
  {
    "age": 35,
    "job": "technician",
    "marital": "single",
    "education": "tertiary",
    "default": "no",
    "balance": 1200.5,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "duration": 210,
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
  }
  ```

- Respuesta 200
  ```json
  {
    "Modelo": "DecisionTreeClassifier",
    "Prediction": "yes" | "no",
    "Probability_yes": 0.0
  }
  ```

- Errores
  - 400: payload inválido o error al predecir.

---

### 3) GET `/api/model/health`
Estado del modelo cargado y rutas de archivos.

- Respuesta 200 (ejemplo)
  ```json
  {
    "source": "file|gridfs|alias|null",
    "path": "artifacts/decision_tree_model.joblib",
    "run_id": "2024-...",
    "pipeline_path": "artifacts/decision_tree_model.joblib",
    "pipeline_exists": true,
    "cache_path": "artifacts/model_cached.joblib",
    "cache_exists": false,
    "loaded": true
  }
  ```

---

### 4) GET `/api/model/refresh`
Fuerza descarga/carga del modelo más reciente desde GridFS o, si no, usa alias local.

- Query params:
  - `run_id` (opcional): fuerza cargar artefacto de un entrenamiento específico.

- Respuesta 200 (ejemplo)
  ```json
  { "status": "ok", "source": "gridfs|alias", "path": "artifacts/model_cached.joblib", "run_id": "..." }
  ```

- Errores
  - 404: no hay runs con artefactos.
  - 500: no se pudo cargar el modelo.

---

### 5) GET `/api/metrics/latest`
Devuelve el registro de métricas más reciente.

- Query params:
  - `include_curves` (bool, default `true`): incluir curvas ROC/PR (pueden ser arrays grandes).

- Respuesta 200: documento completo (sanitizado para JSON). Si `include_curves=false`, el campo `curves` se retorna como `null`.
- Errores
  - 404: no hay métricas registradas.

---

### 6) GET `/api/metrics`
Lista paginada de métricas.

- Query params:
  - `limit` (int, default 10; usa -1 para todos)
  - `page` (int, default 1)
  - `include_curves` (bool, default `false`): si es `false`, se omiten curvas para hacer la respuesta ligera.

- Respuesta 200 (forma)
  ```json
  {
    "total": 12,
    "page": 1,
    "limit": 10,
    "pages": 2,
    "items": [ { "_id": "...", "run_id": "...", "metrics": {"accuracy": 0.85}, "curves": null } ]
  }
  ```

---

### 7) GET `/api/metrics/{run_id}`
Detalle de un run de entrenamiento específico.

- Query params:
  - `include_curves` (bool, default `true`).

- Respuesta 200: documento de ese run (sanitizado para JSON).
- Errores
  - 404: `run_id` no encontrado.

---

## 🧩 Notas técnicas

- El modelo/pipeline se serializa con `joblib`.
- Las métricas y artefactos se leen desde MongoDB/GridFS si está configurado; si no, se usa el modelo local `PIPELINE_PATH`.
- CORS configurable vía `ALLOW_ORIGINS` y `ALLOW_CREDENTIALS`.
- Swagger/Redoc deshabilitados; expón solo los endpoints necesarios.

---

## 🧰 Tecnologías

- **FastAPI** — Framework de la API
- **scikit-learn** — Modelo ML
- **pandas / numpy** — Datos
- **MongoDB + GridFS** — Métricas/artefactos
- **joblib** — Serialización
- **Uvicorn** — Servidor ASGI

---

## ✅ Estado rápido

- Healthcheck: `GET /health` → `{ "status": "ok" }`
- Predicción: `POST /api/predict`
- Modelo: `GET /api/model/health` | `GET /api/model/refresh`
- Métricas: `GET /api/metrics/latest` | `GET /api/metrics` | `GET /api/metrics/{run_id}`

