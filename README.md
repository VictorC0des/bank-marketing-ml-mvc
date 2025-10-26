# 📊 Bank Marketing ML API

API desarrollada con **FastAPI** para predecir si un cliente aceptará una oferta de depósito a plazo, utilizando un modelo de Machine Learning (árbol de decisión) entrenado con el dataset de marketing bancario.

---

## 🚀 Descripción general

Este proyecto entrena un modelo con datos de campañas de marketing bancarias y permite:

- Entrenar y evaluar un modelo con nuevos datos (`train.py`).
- Guardar las métricas generadas en **MongoDB**.
- Realizar predicciones desde la API.
- Consultar las métricas más recientes a través de endpoints REST.
- Probar los endpoints desde **Swagger UI**.

---

## 🧱 Estructura del proyecto

```
bank-marketing-ml-mvc/
│
├── app/
│   ├── controllers/        # Rutas y endpoints (FastAPI)
│   ├── integrations/       # Conexión con MongoDB
│   ├── models/             # Esquemas y carga del modelo
│   ├── main.py             # Punto de entrada de la API
│
├── artifacts/              # Modelos entrenados (.joblib)
│   └── .gitkeep
│
├── scripts/
│   └── train.py            # Entrenamiento y guardado del modelo
│
├── data/                   # Dataset CSV (bank-full.csv)
├── requirements.txt
└── README.md
```

---

## ⚙️ Instalación y ejecución local

### 1️⃣ Clonar el repositorio

```bash
git clone https://github.com/VictorC0des/bank-marketing-ml-mvc.git
cd bank-marketing-ml-mvc
```

### 2️⃣ Crear entorno virtual

```bash
python -m venv venv
venv\Scripts\activate   # En Windows
# o
source venv/bin/activate  # En Linux/Mac
```

### 3️⃣ Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4️⃣ Configurar variables de entorno

Crea un archivo `.env` en la raíz del proyecto con los datos de tu conexión a MongoDB:

```
MONGO_URI=mongodb://localhost:27017
MONGO_DB=bank_marketing
```

---

## 🧠 Entrenamiento del modelo

Ejecuta el script para entrenar y guardar el modelo:

```bash
python scripts/train.py
```

Esto:
- Carga el dataset `data/bank-full.csv`
- Entrena un modelo `DecisionTreeClassifier`
- Guarda el modelo en `artifacts/decision_tree_model.joblib`
- Calcula métricas de rendimiento (Accuracy, F1, ROC AUC, etc.)
- Inserta esas métricas en la base de datos MongoDB.

---

## 🌐 Ejecución de la API

```bash
uvicorn app.main:app --reload
```

Luego abre en tu navegador:

👉 **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📬 Endpoints principales

### 🔹 `POST /api/predict`
Predice si un cliente aceptará la oferta.

**Ejemplo de cuerpo JSON:**
```json
{
  "age": 41,
  "job": "admin.",
  "marital": "married",
  "education": "tertiary",
  "default": "no",
  "balance": 1500,
  "housing": "yes",
  "loan": "no",
  "contact": "cellular",
  "day": 15,
  "month": "may",
  "duration": 200,
  "campaign": 2,
  "pdays": -1,
  "previous": 0,
  "poutcome": "unknown"
}
```

**Respuesta esperada:**
```json
{
  "Modelo": "DecisionTreeClassifier",
  "Prediction": "yes",
  "Probability_yes": 0.74
}
```

---

### 🔹 `GET /api/metrics/latest`
Devuelve las métricas más recientes almacenadas en MongoDB.

**Ejemplo de respuesta:**
```json
{
  "Modelo": "DecisionTreeClassifier",
  "Accuracy": 0.85,
  "Precision": 0.42,
  "Recall": 0.68,
  "F1-Score": 0.52,
  "ROC_AUC": 0.79
}
```

---

## 🧩 Notas técnicas

- El modelo y pipeline se entrenan y guardan con `joblib`.
- Las métricas se guardan automáticamente en MongoDB tras cada entrenamiento.
- Si se vuelve a ejecutar `train.py`, se sobreescribe el modelo anterior.
- Las predicciones usan directamente el pipeline guardado, sin preprocesar manualmente.

---

## 🧰 Tecnologías

- **FastAPI** — Framework para crear la API.
- **scikit-learn** — Entrenamiento y evaluación del modelo.
- **pandas / numpy** — Manipulación de datos.
- **MongoDB** — Almacenamiento de métricas.
- **joblib** — Serialización del modelo.
- **Uvicorn** — Servidor ASGI.

---

## 🔍 Pruebas con Swagger

Puedes probar los endpoints directamente desde Swagger en:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Ahí podrás enviar JSONs de prueba y ver las respuestas del modelo y las métricas.

---
