import argparse
import os
import sys
from pathlib import Path
import json
import datetime
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib

# --- asegurar import del paquete app/* al ejecutar como script ---
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from app.integrations.mongo_repo import save_training_run  # noqa: E402


def downsample(arr, n=200):
    """Reduce longitud de curvas para no saturar Mongo."""
    arr = np.asarray(arr)
    if arr.size <= n:
        return arr.tolist()
    idx = np.linspace(0, arr.size - 1, n).astype(int)
    return arr[idx].tolist()


def load_data(data_path: str, target: str, sep: str | None):
    """Carga CSV con auto-separador si no se especifica; mapea y={yes,no} -> {1,0}."""
    if sep is None:
        # Auto-deteción de separador (soporta ; y ,)
        df = pd.read_csv(data_path, sep=None, engine="python")
    else:
        df = pd.read_csv(data_path, sep=sep)

    if target not in df.columns:
        raise ValueError(f"Columna objetivo '{target}' no encontrada en {data_path}")

    # Mapeo yes/no si aplica
    if df[target].dtype == object:
        df[target] = df[target].map({"yes": 1, "no": 0}).fillna(df[target])

    # Asegurar binaria 0/1
    if df[target].dtype == object:
        # Si seguía siendo string por otros valores, forzamos a 0/1 si procede
        # (asume 1/0 ya está en el CSV; si no, esto lanzará error más adelante si no es binaria)
        pass
    return df


def build_pipeline(cat_cols, num_cols):
    """Crea el preprocesamiento + clasificador en Pipeline."""
    # Para compatibilidad con scikit-learn <1.2 usa sparse=False (sparse_output en 1.2+)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[("ohe", ohe, cat_cols)],
        remainder="passthrough",
    )

    clf = DecisionTreeClassifier(random_state=42)

    pipe = Pipeline(
        steps=[
            ("preproc", preprocessor),
            ("clf", clf),
        ]
    )
    return pipe


def train_and_evaluate(
    df: pd.DataFrame,
    target: str = "y",
    test_size: float = 0.2,
    drop_duration: bool = False,
):
    """Entrena el modelo, calcula métricas y retorna artefactos y documento para Mongo."""
    # Separar X/y
    X = df.drop(columns=[target])
    y = df[target]

    # Evitar leakage si la decisión es previa a la llamada
    if drop_duration and "duration" in X.columns:
        X = X.drop(columns=["duration"])

    # Columnas categóricas y numéricas
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()

    pipe = build_pipeline(cat_cols, num_cols)

    # Stratify si y es binaria
    stratify = y if y.nunique() > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify, random_state=42
    )

    # Grid sencillo (puedes ampliarlo según tiempos)
    param_grid = {
        "clf__criterion": ["gini", "entropy", "log_loss"],
        "clf__max_depth": [3, 5, 7, 9, None],
        "clf__min_samples_split": [2, 10, 20],
        "clf__min_samples_leaf": [1, 10, 50],
        "clf__class_weight": [None, "balanced"],
    }

    gs = GridSearchCV(pipe, param_grid, scoring="f1", cv=5, n_jobs=-1)
    gs.fit(X_train, y_train)

    best = gs.best_estimator_

    # Probabilidades y métricas
    y_prob = best.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    avg_prec = float(average_precision_score(y_test, y_prob))
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Mejor umbral (por F1) para referencia
    thr_grid = np.linspace(0.05, 0.95, 19)
    best_f1, best_t = -1.0, 0.5
    for t in thr_grid:
        y_hat_t = (y_prob >= t).astype(int)
        f1_t = f1_score(y_test, y_hat_t, zero_division=0)
        if f1_t > best_f1:
            best_f1, best_t = float(f1_t), float(t)

    # Curvas (downsample)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    pc, rc, _ = precision_recall_curve(y_test, y_prob)
    curves = {
        "roc_curve": {"fpr": downsample(fpr), "tpr": downsample(tpr)},
        "precision_recall_curve": {"precision": downsample(pc), "recall": downsample(rc)},
    }

    # Nombres de features finales (OHE + numéricas)
    pre = best.named_steps["preproc"]
    ohe = pre.named_transformers_["ohe"]
    try:
        ohe_features = list(ohe.get_feature_names_out(input_features=pre.transformers_[0][2]))
    except Exception:
        # compatibilidad para versiones antiguas de sklearn
        ohe_features = []
        cats = getattr(ohe, "categories_", [])
        cols = pre.transformers_[0][2]
        for i, col in enumerate(cols):
            for v in (cats[i] if i < len(cats) else []):
                ohe_features.append(f"{col}_{v}")
    feature_names = ohe_features + num_cols

    # Documento para Mongo
    doc = {
        "model_name": "DecisionTreeClassifier",
        "model_version": datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
        "params": gs.best_params_,
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "average_precision": avg_prec,
            "confusion_matrix": cm,
            "thresholds": {
                "default": 0.5,
                "best_f1": best_t,
                "f1_at_best_t": best_f1,
            },
        },
        "curves": curves,
        "features_used": feature_names,
        "notes": f"drop_duration={drop_duration}",
        "ts": datetime.datetime.utcnow(),
    }

    return best, doc


def main(
    data_path: str,
    output_dir: str = "artifacts",
    target: str = "y",
    test_size: float = 0.2,
    sep: str | None = None,
    drop_duration: bool = False,
):
    print("📥 Cargando datos:", data_path)
    df = load_data(data_path, target, sep)

    print("🧠 Entrenando modelo (DecisionTree + GridSearchCV)...")
    best_model, mongo_doc = train_and_evaluate(
        df=df, target=target, test_size=test_size, drop_duration=drop_duration
    )

    # Guardar modelo
    artifacts = Path(output_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    model_path = artifacts / "decision_tree_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"💾 Modelo guardado en: {model_path}")

    # Guardar métricas en Mongo
    run_id = save_training_run(mongo_doc)
    print("✅ Métricas insertadas en MongoDB (training_runs).")
    print(f"   run_id: {run_id}")
    print("📊 Resumen:")
    m = mongo_doc["metrics"]
    print(
        json.dumps(
            {
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "roc_auc": m["roc_auc"],
                "average_precision": m["average_precision"],
                "best_threshold": m["thresholds"]["best_f1"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena DecisionTree y guarda métricas en MongoDB.")
    parser.add_argument("--data-path", type=str, default="data/bank-full.csv", help="Ruta del CSV de entrenamiento.")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Carpeta para guardar el modelo.")
    parser.add_argument("--target", type=str, default="y", help="Nombre de la columna objetivo.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporción para test.")
    parser.add_argument(
        "--sep",
        type=str,
        default=None,
        help='Separador del CSV. Usa None para autodetectar (; o ,).',
    )
    parser.add_argument(
        "--drop-duration",
        action="store_true",
        help="Si se pasa, elimina la columna 'duration' para evitar leakage.",
    )
    args = parser.parse_args()

    main(
        data_path=args.data_path,
        output_dir=args.output_dir,
        target=args.target,
        test_size=args.test_size,
        sep=args.sep,
        drop_duration=args.drop_duration,
    )
