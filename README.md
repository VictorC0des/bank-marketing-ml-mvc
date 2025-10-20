# 🧠 Bank Marketing Decision Tree API

Este proyecto implementa un **modelo de Árbol de Decisión** para predecir si un cliente **aceptará una oferta bancaria** (depósito a plazo) basándose en el famoso dataset **Bank Marketing (UCI)**.  
Incluye un pipeline completo de **entrenamiento, evaluación, almacenamiento de métricas y API REST** para servir predicciones y resultados del modelo.

---

## 📘 Contexto

El propósito es ayudar a un banco a **decidir a quién contactar** en campañas telefónicas para mejorar la tasa de éxito.  
El modelo clasifica clientes en “**sí**” (aceptará la oferta) o “**no**” (no aceptará), basándose en sus datos personales y de contacto.

El sistema:
- Entrena un modelo de Machine Learning (Decision Tree).
- Calcula métricas de desempeño (Accuracy, Precision, Recall, F1, ROC AUC, Curvas, Matriz de confusión).
- Expone una **API con FastAPI** para consultar predicciones y métricas.

---

## ⚙️ Estructura del proyecto

```
bank-marketing-decisiontree-api/
│
├── app/
│   ├── api.py              # Rutas principales de la API (endpoints)
│   ├── main.py             # Punto de entrada de FastAPI
│   ├── response.py         # Estructura de respuestas JSON
│
├── scripts/
│   ├── train.py            # Entrenamiento del modelo y guardado de métricas
│
├── artifacts/
│   ├── decision_tree_model.joblib   # Modelo entrenado
│   ├── metrics.json                 # Métricas del modelo
│   ├── curves.json                  # Curvas ROC y Precision-Recall
│
├── data/
│   └── bank.csv             # Dataset original (UCI Bank Marketing)
│
├── requirements.txt          # Dependencias del proyecto
└── README.md                 # Este archivo
```

---

## 🚀 Ejecución paso a paso

### 🔹 1. Clonar el repositorio

```bash
git clone https://github.com/usuario/bank-marketing-decisiontree-api.git
cd bank-marketing-decisiontree-api
```

> Reemplaza `usuario` por el nombre de tu repositorio o descarga el ZIP y descomprímelo.

---

### 🔹 2. Crear entorno virtual

```bash
python -m venv venv
```

Activar entorno:

- **Windows:**
  ```bash
  venv\Scripts\activate
  ```

- **Mac/Linux:**
  ```bash
  source venv/bin/activate
  ```

---

### 🔹 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

### 🔹 4. Entrenar el modelo

Ejecuta el script de entrenamiento.  
Esto generará los archivos `decision_tree_model.joblib`, `metrics.json` y `curves.json` en la carpeta `artifacts/`.

```bash
python scripts/train.py
```

Si todo sale bien, verás un mensaje como:
```
✅ Modelo entrenado y guardado correctamente en artifacts/
```

---

### 🔹 5. Ejecutar la API

Lanza el servidor FastAPI:

```bash
uvicorn app.main:app --reload
```

Luego abre en el navegador:

👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Ahí podrás probar todos los endpoints con una interfaz interactiva.

---

## 🌐 Endpoints principales

| Endpoint | Método | Descripción |
|-----------|--------|--------------|
| `/api/predict` | POST | Predice si un cliente aceptará la oferta |
| `/api/metrics` | GET | Retorna las métricas, matriz de confusión y curvas |
| `/health` | GET | Comprueba si la API está funcionando |

---

### 🔸 Ejemplo de predicción

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/api/predict" -H "Content-Type: application/json" -d '{
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
}'
```

**Response:**
```json
{
  "Prediction": "no",
  "Probability_yes": 0.23
}
```

---

### 🔸 Ejemplo de métricas

**Request:**
```bash
curl http://127.0.0.1:8000/api/metrics
```

**Response (ejemplo):**
```json
{
  "Modelo": "DecisionTreeClassifier",
  "Accuracy": 0.853,
  "Precision": 0.422,
  "Recall": 0.690,
  "F1-Score": 0.524,
  "ROC_AUC": 0.800,
  "Matriz_de_Confusion": [[6985,1000],[328,730]]
}
```

---

## 🧩 Explicación de métricas

- **Accuracy:** Porcentaje total de aciertos.  
- **Precision:** Qué proporción de las predicciones “sí” fueron correctas.  
- **Recall:** Qué proporción de los verdaderos “sí” fueron detectados.  
- **F1-Score:** Balance entre precision y recall.  
- **ROC_AUC:** Capacidad del modelo para separar clases (0.5=aleatorio, 1.0=perfecto).  
- **Matriz de confusión:** Muestra aciertos y errores por tipo de clase.

---

## 🖥️ Despliegue en servidor remoto

Para ejecutar en otra máquina (por ejemplo, un servidor Linux o nube):
1. Clonar el repositorio igual que antes.
2. Instalar Python 3.9+ y dependencias.
3. Activar entorno virtual.
4. Ejecutar `train.py` (si aún no existen los archivos en `artifacts/`).
5. Lanzar la API con:
   ```bash
   nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 &
   ```
6. Acceder desde cualquier navegador en:  
   `http://<IP_DEL_SERVIDOR>:8000/docs`

---

## 📊 Resultados del modelo

| Métrica | Valor |
|----------|--------|
| Accuracy | 0.853 |
| Precision | 0.422 |
| Recall | 0.690 |
| F1-Score | 0.524 |
| ROC_AUC | 0.800 |

El modelo muestra **buen recall** (detecta la mayoría de los clientes que sí aceptarían), aunque la precision es moderada (algunos falsos positivos).

---

## 🔍 Mejoras posibles

- Ajustar **umbral de decisión** según la estrategia (más precisión o más recall).
- Probar otros modelos (RandomForest, XGBoost).
- Eliminar la variable `duration` si el modelo se usará *antes* de hacer llamadas (para evitar data leakage).
- Implementar logging y monitoreo en producción.

---

## 👨‍💻 Autor
**Equipo de Data Science – Bank Marketing Project**  
Proyecto académico para la materia de *Aprendizaje Automático*.

---

## 🏁 Licencia
Uso académico y educativo. Libre para replicar y modificar con fines didácticos.
