# 📊 Bank Marketing ML API — Documentación de Endpoints

API REST desarrollada con **FastAPI** que permite predecir si un cliente aceptará un depósito a plazo bancario.  
Incluye dos modelos:

- `decision_tree` — Árbol de decisión (Machine Learning clásico)
- `deep_learning` — Red neuronal (Deep Learning)

La API también expone endpoints para consultar métricas, refrescar modelos y revisar el estado de los artefactos.

---

# 🌐 URL Base

### Producción
```
https://bank-marketing-ml-mvc.onrender.com
```

### Desarrollo (opcional)
```
http://localhost:8000
```

---

# 🧾 Esquema del Body para Predicciones

Todos los requests de predicción usan este formato:

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

---

# 📚 Referencia Completa de Endpoints

---

# 1️⃣ Healthcheck

## `GET /health`

Verifica que la API está activa.

Ejemplo:
```
GET https://bank-marketing-ml-mvc.onrender.com/health
```

Respuesta:
```json
{ "status": "ok" }
```

---

# 2️⃣ Predicción

## `POST /api/predict`

Realiza una predicción usando el modelo seleccionado.

### Query Params

| Parámetro     | Tipo   | Default         | Valores permitidos                  |
|---------------|--------|------------------|--------------------------------------|
| model_type    | string | decision_tree    | decision_tree, deep_learning         |

### Body
(ver sección previa)

### Ejemplos:

Decision Tree:
```
POST https://bank-marketing-ml-mvc.onrender.com/api/predict?model_type=decision_tree
```

Deep Learning:
```
POST https://bank-marketing-ml-mvc.onrender.com/api/predict?model_type=deep_learning
```

---

# 3️⃣ Estado de los Modelos

## `GET /api/model/health`

Retorna la información del estado actual de los modelos cargados.

Ejemplo:
```
GET https://bank-marketing-ml-mvc.onrender.com/api/model/health
```

---

# 4️⃣ Recargar Modelos

## `GET /api/model/refresh`

Fuerza la recarga del modelo desde almacenamiento en GridFS o artefactos locales.

### Query Params

| Parámetro     | Tipo   | Obligatorio |
|---------------|--------|-------------|
| model_type    | string | Sí          |
| run_id        | string | No          |

Ejemplos:

```
GET https://bank-marketing-ml-mvc.onrender.com/api/model/refresh?model_type=decision_tree
```

```
GET https://bank-marketing-ml-mvc.onrender.com/api/model/refresh?model_type=deep_learning&run_id=2024-11-29T10:30:00
```

---

# 5️⃣ Últimas Métricas

## `GET /api/metrics/latest`

Devuelve las métricas más recientes.

### Query Params

| Parámetro        | Tipo   | Default |
|------------------|--------|---------|
| model_type       | string | null    |
| include_curves   | bool   | true    |

Ejemplos:

```
GET https://bank-marketing-ml-mvc.onrender.com/api/metrics/latest?include_curves=false
```

```
GET https://bank-marketing-ml-mvc.onrender.com/api/metrics/latest?model_type=deep_learning
```

---

# 6️⃣ Lista paginada de métricas

## `GET /api/metrics`

Lista registros de métricas con paginación.

### Query Params

| Parámetro        | Tipo   | Default |
|------------------|--------|---------|
| model_type       | string | null    |
| limit            | int    | 10      |
| page             | int    | 1       |
| include_curves   | bool   | false   |

Ejemplos:

```
GET https://bank-marketing-ml-mvc.onrender.com/api/metrics
```

```
GET https://bank-marketing-ml-mvc.onrender.com/api/metrics?model_type=decision_tree&limit=20&page=2
```

```
GET https://bank-marketing-ml-mvc.onrender.com/api/metrics?model_type=deep_learning&limit=-1&include_curves=true
```

---

# 7️⃣ Métricas por run_id

## `GET /api/metrics/{run_id}`

Devuelve las métricas asociadas a un entrenamiento específico.

Ejemplo:
```
GET https://bank-marketing-ml-mvc.onrender.com/api/metrics/2024-11-29T10:30:00?include_curves=false
```

---

# 🧰 Tecnologías

- FastAPI
- scikit-learn
- TensorFlow/Keras
- MongoDB + GridFS
- Joblib / H5
- Uvicorn

---

# ✅ Resumen de Endpoints

| Endpoint                      | Método | Descripción                            |
|------------------------------|--------|----------------------------------------|
| `/health`                    | GET    | Estado del servicio                    |
| `/api/predict`               | POST   | Predicción (DT o DL)                   |
| `/api/model/health`          | GET    | Estado de modelos                      |
| `/api/model/refresh`         | GET    | Recargar modelo                        |
| `/api/metrics/latest`        | GET    | Últimas métricas                       |
| `/api/metrics`               | GET    | Métricas paginadas                     |
| `/api/metrics/{run_id}`      | GET    | Métrica específica                      |
