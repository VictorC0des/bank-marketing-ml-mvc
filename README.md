# 📊 Bank Marketing — Single Decision Tree Trainer

Entrenador de un único árbol de decisión (sin ensembles) para el dataset clásico de marketing bancario. Guarda el pipeline calibrado, registra métricas en MongoDB y prioriza reproducibilidad.

---

## 🚀 ¿Qué incluye?

- Entrenamiento de un único `DecisionTreeClassifier` con One-Hot Encoder de categorías infrecuentes.
- Tuning activado por defecto (búsqueda aleatoria 80×5) para mejorar el ranking de probabilidades.
- Calibración por defecto con `sigmoid` para probabilidades más útiles.
- Umbralización opcional con piso de precisión (útil en modo operativo).
- Registro de métricas en **MongoDB** y guardado del artefacto en `artifacts/`.

---

## 🧱 Estructura del proyecto

```
bank-marketing-ml-mvc/
├── artifacts/                 # Modelos entrenados (.joblib, con timestamp)
├── data/
│   └── bank-full.csv          # Dataset de entrenamiento (separado por ';')
├── integrations/
│   ├── featurize.py           # Ingeniería de variables
│   └── mongo_repo.py          # Registro de métricas en MongoDB
├── scripts/
│   └── train.py               # Entrenamiento principal (CLI)
├── requirements.txt
└── README.md
```

---

## ⚙️ Instalación (Windows PowerShell)

1) Crear entorno y activarlo

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Instalar dependencias

```
python -m pip install -r requirements.txt
```

3) Variables de entorno (MongoDB)

Crea un archivo `.env` en la raíz del proyecto:

```
MONGO_URI=mongodb://localhost:27017
MONGO_DB=bank_marketing
```

---

## 🧠 Entrenamiento rápido

Con los valores por defecto dejamos listo un modo enfocado a reproducibilidad y buen AP (~0.55 con un solo DT), sin pasar flags:

```
python scripts/train.py
```

Esto hará:
- Cargar `data/bank-full.csv` (separador `;`).
- Aplicar ingeniería de variables.
- Hacer tuning 80×5 del árbol.
- Calibrar probabilidades con `sigmoid`.
- Guardar el pipeline en `artifacts/DecisionTree_YYYYMMDD_HHMMSS.joblib`.
- Registrar métricas en MongoDB.

Métricas esperadas con un único DT (aprox.):
- average_precision ≈ 0.54–0.56
- precision/recall dependerán del umbral óptimo interno (optimize=f1 por defecto).

Nota: Por diseño, un solo árbol no alcanza AP≥0.60 de forma robusta en este dataset sin incurrir en leakage o ensembles.

---

## � Modos de operación

### 1) Modo AP puro (por defecto)

- tuning activado (80×5)
- calibración: `sigmoid`
- sin re-muestreo de clases
- optimize: `f1`

Ejecuta sin flags:

```
python scripts/train.py
```

### 2) Modo operativo con piso de precisión (≥ 0.62)

Maximiza el recall sujeto a una precisión mínima, útil cuando los falsos positivos tienen mayor costo.

Ejemplo recomendado:

```
python scripts/train.py --optimize recall --min-precision 0.62 --resample over --resample-ratio 0.5
```

Observaciones típicas (aprox.):
- precision ≈ 0.62–0.63
- recall ≈ 0.35–0.40
- average_precision ≈ 0.54–0.55

### 3) Evitar leakage por `duration`

`duration` sólo se conoce después de la llamada; para una evaluación realista:

```
python scripts/train.py --drop-duration
```

---

## � Parámetros principales (CLI)

- `--optimize {f1,fbeta,precision,recall,cost}`: objetivo de umbralización (por defecto: f1).
- `--min-precision FLOAT`: piso de precisión (omitir por defecto).
- `--tune-dt` (activado por defecto): habilita tuning del árbol.
- `--tune-iter INT` (defecto: 80) y `--tune-folds INT` (defecto: 5).
- `--calibration {sigmoid,isotonic,none}` (defecto: sigmoid).
- `--resample {none,over,under,smote}` (defecto: none) y `--resample-ratio FLOAT` (defecto: 0.5).
- `--drop-duration`: elimina columnas de duración (y derivadas) para evitar leakage.
- `--no-feat`: desactiva la ingeniería de variables interna.

---

## 🧪 Reproducibilidad

- `random_state=42` por defecto y particiones estratificadas.
- El encoder agrupa categorías infrecuentes para reducir ruido en el ranking.
- Calibración separada en hold-out para mejorar las probabilidades.

---

## 📌 Notas y límites conocidos

- Requisito cumplido: sólo se usa un `DecisionTreeClassifier` (sin ensembles).
- En este dataset, AP ≈ 0.55 es un techo razonable con un único árbol y sin leakage.
- Ensembles (RF/GBM/XGB) mejoran AP, pero no se usan para cumplir la restricción del profesor.

---

## 🗺️ Siguientes pasos (opcionales)

- Exportar el árbol a Graphviz para el informe:

```
# Ejemplo rápido
from sklearn import tree
import joblib
model = joblib.load('artifacts/DecisionTree_YYYYMMDD_HHMMSS.joblib')
dt = model.named_steps['model'].base_estimator if hasattr(model, 'named_steps') else model
# Si calibrado, extraer el estimator subyacente
if hasattr(dt, 'calibrated_classifiers_'):
    dt = dt.calibrated_classifiers_[0].estimator
tree.export_graphviz(dt, out_file='tree.dot', filled=True, feature_names=None)
```

---

## 🧰 Tecnologías

- scikit-learn, pandas, numpy, scipy, joblib
- imbalanced-learn
- pymongo, python-dotenv

