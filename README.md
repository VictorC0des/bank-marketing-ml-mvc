# 📊 Bank Marketing ML API

API de predicción basada en **FastAPI** para estimar si un cliente aceptará una oferta de depósito a plazo, usando modelos de **Machine Learning** (árbol de decisión y redes neuronales) entrenados con el dataset de marketing bancario.

---

## 🚀 Descripción general

Este proyecto permite:

- **Exponer endpoints REST** para consultar métricas, estado del modelo y realizar predicciones.
- **Realizar predicciones en línea** mediante modelos cargados (Decision Tree + Deep Learning).
- **Leer y servir métricas** almacenadas en MongoDB (si está configurado con variables `MONGO_*`).

> ⚠️ **Importante:** La documentación interactiva (Swagger/Redoc) está deshabilitada en producción. Usa **Postman** o **curl** para consumir la API.

---

## 🧱 Estructura del proyecto

```
bank-marketing-ml-mvc/
│
├── app/
│   ├── controllers/
│   │   └── api.py              # Endpoints de la API (prefijo /api)
│   ├── models/
│   │   └── pipeline.py         # Carga/refresh del modelo y predicción
│   ├── views/
│   │   └── responses.py        # Esquemas de request/response
│   └── main.py                 # FastAPI app, CORS y /health
│
├── integrations/
│   ├── featurize.py            # Transformación de features
│   └── mongo_repo.py           # Acceso a MongoDB/GridFS
│
├── artifacts/
│   ├── decision_tree_model.joblib
│   └── deep_learning_model.h5
│
├── data/
│   └── bank-full.csv           # Dataset de referencia
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuración (variables de entorno)

Copia `.env.example` (o crea `.env`) con estas variables:

```bash
# === MONGODB ===
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/?
MONGO_DB=bank_ml

# === MODELOS (Machine Learning - Decision Tree) ===
PIPELINE_PATH=artifacts/decision_tree_model.joblib
MODEL_CACHE_PATH=artifacts/model_cached.joblib

# === MODELOS (Deep Learning - Neural Network) ===
DEEPLEARNING_MODEL_PATH=artifacts/deep_learning_model.h5
DEEPLEARNING_MODEL_CACHE_PATH=artifacts/dl_model_cached.h5
DEEPLEARNING_SCALER_PATH=artifacts/dl_scaler.joblib
DEEPLEARNING_SCALER_CACHE_PATH=artifacts/dl_scaler_cached.joblib
DEEPLEARNING_ENCODER_PATH=artifacts/dl_encoder.joblib
DEEPLEARNING_ENCODER_CACHE_PATH=artifacts/dl_encoder_cached.joblib

# === CORS ===
ALLOW_ORIGINS=*
ALLOW_CREDENTIALS=false
```

**Variables clave:**

| Variable | Requerida | Descripción |
|----------|-----------|-------------|
| `MONGO_URI` | ✅ | URI de conexión a MongoDB (GridFS, métricas) |
| `MONGO_DB` | ✅ | Base de datos con colección `training_runs` |
| `PIPELINE_PATH` | ✅ | Ruta del modelo Decision Tree (fallback local) |
| `DEEPLEARNING_*` | ✅ | Rutas de modelo, scaler y encoder DL |
| `ALLOW_ORIGINS` | ✅ | Orígenes CORS (`*` = desarrollo; especificar en prod) |

---

## 🌐 Ejecutar la API

### Local (hot-reload)

```bash
uvicorn app.main:app --reload
```

### Docker

```bash
docker build -t bank-api:latest .
docker run --rm -p 8000:8000 --env-file .env bank-api:latest
```

**Base URL:** `http://localhost:8000`

---

## 📚 Referencia de la API

Prefijo común: `/api` (excepto `/health`).

### 1️⃣ `GET /health`

**Ping del servicio.**

```bash
curl http://localhost:8000/health
```

**Respuesta 200:**
```json
{ "status": "ok" }
```

---

### 2️⃣ `POST /api/predict`

**Predice si el cliente aceptará la oferta.**

**Body (JSON) — Esquema InputData:**

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

**Query params:**

| Param | Tipo | Default | Valores |
|-------|------|---------|---------|
| `model_type` | str | `decision_tree` | `decision_tree` \| `deep_learning` |

**Ejemplo:**

```bash
curl -X POST http://localhost:8000/api/predict?model_type=decision_tree \
  -H "Content-Type: application/json" \
  -d @payload.json
```

**Respuesta 200:**
```json
{
  "Modelo": "DecisionTreeClassifier",
  "model_type": "decision_tree",
  "Prediction": "yes",
  "Probability_yes": 0.72
}
```

**Errores:**
- `400`: Payload inválido o error al predecir.

---

### 3️⃣ `GET /api/model/health`

**Estado del modelo cargado y rutas de archivos.**

```bash
curl http://localhost:8000/api/model/health
```

**Respuesta 200:**
```json
{
  "status": "ok",
  "models": {
    "decision_tree": {
      "source": "file",
      "path": "artifacts/decision_tree_model.joblib",
      "run_id": "2024-11-29T10:30:00",
      "pipeline_path": "artifacts/decision_tree_model.joblib",
      "pipeline_exists": true,
      "cache_path": "artifacts/model_cached.joblib",
      "cache_exists": false,
      "loaded": true
    },
    "deep_learning": {
      "model_path": "artifacts/deep_learning_model.h5",
      "model_exists": true,
      "scaler_path": "artifacts/dl_scaler.joblib",
      "scaler_exists": true,
      "encoder_path": "artifacts/dl_encoder.joblib",
      "encoder_exists": true,
      "model_loaded": true,
      "scaler_loaded": true,
      "encoder_loaded": true
    }
  }
}
```

---

### 4️⃣ `GET /api/model/refresh`

**Fuerza descarga/carga del modelo más reciente desde MongoDB/GridFS.**

**Query params:**

| Param | Tipo | Default | Descripción |
|-------|------|---------|-------------|
| `model_type` | str | `decision_tree` | Tipo a refrescar: `decision_tree` \| `deep_learning` |
| `run_id` | str | `null` | Fuerza cargar artefacto de un training específico |

**Ejemplos:**

```bash
# Refrescar Decision Tree
curl http://localhost:8000/api/model/refresh?model_type=decision_tree

# Refrescar Deep Learning desde un run específico
curl http://localhost:8000/api/model/refresh?model_type=deep_learning&run_id=abc123
```

**Respuesta 200:**
```json
{
  "status": "ok",
  "model_type": "decision_tree",
  "source": "gridfs",
  "path": "artifacts/model_cached.joblib",
  "run_id": "2024-11-29T10:30:00"
}
```

**Errores:**
- `404`: No hay runs con artefactos.
- `500`: No se pudo cargar el modelo.

---

### 5️⃣ `GET /api/metrics/latest`

**Devuelve el registro de métricas más reciente.**

**Query params:**

| Param | Tipo | Default | Descripción |
|-------|------|---------|-------------|
| `model_type` | str | `null` | Filtrar por: `decision_tree` \| `deep_learning` (opcional) |
| `include_curves` | bool | `true` | Incluir curvas ROC/PR (pueden ser arrays grandes) |

**Ejemplos:**

```bash
# Últimas métricas sin curvas
curl "http://localhost:8000/api/metrics/latest?include_curves=false"

# Últimas métricas de Decision Tree
curl "http://localhost:8000/api/metrics/latest?model_type=decision_tree&include_curves=true"
```

**Respuesta 200:** Documento completo (sanitizado para JSON). Si `include_curves=false`, el campo `curves` es `null`.

```json
{
  "_id": "abc123xyz",
  "run_id": "2024-11-29T10:30:00",
  "model_type": "decision_tree",
  "model_name": "DecisionTreeClassifier",
  "metrics": {
    "accuracy": 0.89,
    "precision": 0.87,
    "recall": 0.85,
    "f1": 0.86
  },
  "curves": null,
  "timestamp": "2024-11-29T10:30:00"
}
```

**Errores:**
- `404`: No hay métricas registradas.

---

### 6️⃣ `GET /api/metrics`

**Lista paginada de métricas.**

**Query params:**

| Param | Tipo | Default | Descripción |
|-------|------|---------|-------------|
| `model_type` | str | `null` | Filtrar por tipo (opcional) |
| `limit` | int | `10` | Resultados por página; `-1` = todos |
| `page` | int | `1` | Número de página |
| `include_curves` | bool | `false` | Incluir curvas (false = respuesta ligera) |

**Ejemplos:**

```bash
# Primeros 10 registros sin curvas
curl "http://localhost:8000/api/metrics"

# Página 2, 20 resultados, Decision Tree
curl "http://localhost:8000/api/metrics?model_type=decision_tree&limit=20&page=2"

# Todos los registros de Deep Learning con curvas
curl "http://localhost:8000/api/metrics?model_type=deep_learning&limit=-1&include_curves=true"
```

**Respuesta 200:**
```json
{
  "total": 62,
  "page": 1,
  "limit": 10,
  "pages": 7,
  "items": [
    {
      "_id": "abc123",
      "run_id": "2024-11-29T10:30:00",
      "model_type": "decision_tree",
      "metrics": { "accuracy": 0.89 },
      "curves": null
    }
  ]
}
```

**Errores:**
- `404`: No hay métricas para ese filtro.

---

### 7️⃣ `GET /api/metrics/{run_id}`

**Detalle de un training específico.**

**Path params:**
- `run_id` (str): ID único del training run

**Query params:**

| Param | Tipo | Default | Descripción |
|-------|------|---------|-------------|
| `include_curves` | bool | `true` | Incluir curvas ROC/PR |

**Ejemplo:**

```bash
curl "http://localhost:8000/api/metrics/2024-11-29T10:30:00?include_curves=true"
```

**Respuesta 200:** Documento sanitizado para JSON.

**Errores:**
- `404`: `run_id` no encontrado.

---

## 🧩 Notas técnicas

- **Serialización:** El modelo/pipeline se serializa con **joblib** (Decision Tree) o **h5** (Deep Learning).
- **Almacenamiento:** Métricas y artefactos se leen desde **MongoDB/GridFS** si está configurado; fallback a modelo local `PIPELINE_PATH`.
- **CORS:** Configurable vía `ALLOW_ORIGINS` y `ALLOW_CREDENTIALS`.
- **Documentación:** Swagger/Redoc deshabilitados; expón solo endpoints necesarios.
- **Dual Model:** Sistema carga ambos modelos en startup; endpoint `/api/predict` elige cuál usar vía `model_type`.
- **GridFS Fallback:** Si el artefacto no existe en archivo local, la API intenta descargarlo de GridFS automáticamente.

---

## 🧰 Tecnologías

- **FastAPI** — Framework REST moderno
- **scikit-learn** — Árbol de decisión (ML)
- **TensorFlow/Keras** — Redes neuronales (DL)
- **pandas / numpy** — Transformación de datos
- **MongoDB + GridFS** — Métricas y almacenamiento de artefactos
- **joblib** — Serialización de modelos
- **Uvicorn** — Servidor ASGI

---

## ✅ Estado rápido

| Endpoint | Método | Propósito |
|----------|--------|----------|
| `/health` | GET | Healthcheck |
| `/api/predict` | POST | Predicción (Decision Tree o Deep Learning) |
| `/api/model/health` | GET | Estado de modelos |
| `/api/model/refresh` | GET | Refrescar modelo desde MongoDB |
| `/api/metrics/latest` | GET | Última métrica |
| `/api/metrics` | GET | Lista paginada de métricas |
| `/api/metrics/{run_id}` | GET | Detalle de training específico |

---

**🚀 Listo para producción. Usa Postman, curl o tu cliente HTTP favorito.**

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
- Predicción: `POST /api/predict?model_type=decision_tree` | `POST /api/predict?model_type=deep_learning`
- Modelo: `GET /api/model/health` | `GET /api/model/refresh`
- Métricas: `GET /api/metrics/latest` | `GET /api/metrics` | `GET /api/metrics/{run_id}`

---

## 📋 Frontend Compatibility

**Para frontend (React/Vue/Angular):**

- **Base URL**: `http://localhost:8000` (o tu servidor)
- **Endpoints principales**:
  - `GET /health` → verifica si API está viva
  - `POST /api/predict?model_type=decision_tree` → predicción con DT
  - `POST /api/predict?model_type=deep_learning` → predicción con DL
  - `GET /api/metrics/latest?model_type=deep_learning` → última métrica DL
  - `GET /api/metrics?model_type=decision_tree&limit=5` → historial últimos 5 DT

**Headers recomendados**:
```javascript
fetch('/api/predict?model_type=decision_tree', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({...})
})
```

**Cambios desde v1:**
- ✅ **NO BREAKING CHANGES** — el parámetro `?model_type=` es opcional
- Frontend puede usar `decision_tree` por defecto (backward-compatible)
- O agregar UI para elegir modelo
- O llamar ambos en paralelo para comparar

---

## 🎯 Resumen de cambios desde v1

| Aspecto | v1 (Decision Tree solo) | v2 (Dual Model) |
|---|---|---|
| **Modelos soportados** | 1 (DT) | 2 (DT + DL) |
| **Endpoints** | `/api/predict` | `/api/predict?model_type=` |
| **Métricas** | Global | Filtrable por `?model_type=` |
| **GridFS** | Solo modelo | Modelo + preprocessor |
| **Model detection** | Por ID de run | Por nombre + ID de run |
| **Backward compat** | N/A | ✅ Docs antiguos soportados |
| **Frontend breaking** | N/A | ❌ NO (default = decision_tree) |

