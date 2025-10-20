# 🧠 Bank Marketing Decision Tree - API

Este proyecto implementa un **modelo de Árbol de Decisión** entrenado con el dataset **Bank Marketing (UCI)**.  
Su propósito es predecir si un cliente **aceptará o no** una oferta de depósito a plazo, utilizando un conjunto de variables socioeconómicas y de contacto.

El modelo se entrena mediante un **pipeline de Machine Learning** y se despliega mediante una **API con FastAPI** para exponer resultados y métricas en formato JSON.

---

## ⚙️ Estructura del proyecto

```
bank-marketing-ml-mvc/
│
├── app/                    # API con FastAPI
│   ├── main.py             # Punto de entrada de la aplicación
│   ├── controllers/        # Endpoints y rutas
│   ├── models/             # Modelo y lógica de predicción
│   ├── views/              # Respuestas estructuradas
│
├── artifacts/              # Resultados del entrenamiento
│   ├── decision_tree_model.joblib
│   ├── metrics.json
│   ├── curves.json
│
├── data/
│   └── bank-full.csv       # Dataset original
│
├── scripts/
│   └── train.py            # Script de entrenamiento del modelo
│
├── tests/
│   └── test_smoke.py       # Pruebas básicas del API
│
├── Dockerfile              # Configuración para despliegue
├── requirements.txt        # Dependencias
└── README.md               # Documentación
```

---

## 🚀 Ejecución paso a paso

### 1️⃣ Clonar el repositorio

```bash
git clone https://github.com/VictorC0des/bank-marketing-ml-mvc.git
cd bank-marketing-ml-mvc
```

---

### 2️⃣ Crear y activar el entorno virtual

```bash
python -m venv venv
```

Activar entorno virtual:

- **Windows**
  ```bash
  venv\Scripts\activate
  ```
- **Mac/Linux**
  ```bash
  source venv/bin/activate
  ```

---

### 3️⃣ Instalar dependencias

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Entrenar el modelo

Ejecuta el script de entrenamiento con el dataset:

```bash
python scripts/train.py
```

Esto generará los siguientes archivos dentro de `artifacts/`:
- `decision_tree_model.joblib` → modelo entrenado
- `metrics.json` → métricas principales (Accuracy, Recall, Precision, etc.)
- `curves.json` → curvas ROC y Precision-Recall

---

### 5️⃣ Ejecutar la API

Inicia el servidor local con:

```bash
uvicorn app.main:app --reload
```

Por defecto, el servidor se iniciará en:

👉 **http://127.0.0.1:8000**

---

## 🌐 Probar endpoints en Swagger

FastAPI genera automáticamente la documentación interactiva.

Abre en tu navegador:

👉 **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

Ahí podrás probar todos los endpoints, incluyendo:
- `/api/predict` → para realizar predicciones
- `/api/metrics` → para visualizar métricas del modelo
- `/health` → para verificar el estado del servicio

---

## 📊 Ejemplo de uso del endpoint `/api/predict`

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/api/predict" \
-H "Content-Type: application/json" \
-d '{
  "age": 35,
  "job": "management",
  "marital": "married",
  "education": "tertiary",
  "default": "no",
  "balance": 1500,
  "housing": "yes",
  "loan": "no",
  "contact": "cellular",
  "day": 10,
  "month": "may",
  "duration": 120,
  "campaign": 2,
  "pdays": -1,
  "previous": 0,
  "poutcome": "unknown"
}'
```

**Response:**
```json
{
  "Prediction": "no",
  "Probability_yes": 0.27
}
```

---

## 📈 Ejemplo del endpoint `/api/metrics`

**Request:**
```bash
curl http://127.0.0.1:8000/api/metrics
```

**Response:**
```json
{
  "Modelo": "DecisionTreeClassifier",
  "Accuracy": 0.85,
  "Precision": 0.42,
  "Recall": 0.69,
  "F1-Score": 0.52,
  "ROC_AUC": 0.80,
  "Matriz_de_Confusion": [[6985, 1000], [328, 730]]
}
```

---

## ✅ Resumen

Este proyecto permite:

- Entrenar y guardar un modelo de clasificación basado en Árboles de Decisión.  
- Consultar sus métricas de rendimiento.  
- Exponer predicciones y resultados mediante una API accesible vía Swagger.
