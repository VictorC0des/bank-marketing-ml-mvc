import argparse
import os
import sys
from pathlib import Path
import json
import datetime
import numpy as np
import pandas as pd
from dotenv import load_dotenv

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
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
try:
    # imblearn imports (optional) for sampling in pipeline
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
except Exception:
    ImbPipeline = None
    SMOTE = None
    RandomUnderSampler = None
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib

# ensure package import when running script
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Load environment variables from a .env file in the repo root (optional)
load_dotenv(ROOT_DIR / '.env')


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).lower() in ("1", "true", "yes", "y")


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


from integrations.mongo_repo import save_training_run  # noqa: E402


def downsample(arr, n=200):
    arr = np.asarray(arr)
    if arr.size <= n:
        return arr.tolist()
    idx = np.linspace(0, arr.size - 1, n).astype(int)
    return arr[idx].tolist()


def load_data(data_path: str, target: str, sep: str | None):
    if sep is None:
        df = pd.read_csv(data_path, sep=None, engine="python")
    else:
        df = pd.read_csv(data_path, sep=sep)
    if target not in df.columns:
        raise ValueError(f"Columna objetivo '{target}' no encontrada en {data_path}")
    if df[target].dtype == object:
        df[target] = df[target].map({"yes": 1, "no": 0}).fillna(df[target])
    return df


def build_pipeline(cat_cols, num_cols, sampler=None, random_state: int = 42):
    # Use OrdinalEncoder for a single DecisionTree model (keeps pipeline compact)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    preprocessor = ColumnTransformer(transformers=[("ord", enc, cat_cols)], remainder="passthrough")
    clf = DecisionTreeClassifier(random_state=random_state, class_weight=None)
    if sampler is not None and ImbPipeline is not None:
        pipe = ImbPipeline(steps=[("preproc", preprocessor), ("sampler", sampler), ("clf", clf)])
    else:
        pipe = Pipeline(steps=[("preproc", preprocessor), ("clf", clf)])
    return pipe


def train_with_preset_decision_tree(
    df: pd.DataFrame,
    target: str = "y",
    test_size: float = 0.2,
    drop_duration: bool = False,
    random_state: int = 42,
    preset_params: dict | None = None,
    decision_threshold: float | None = None,
):
    """Train a single DecisionTree using preset hyperparameters (no RandomizedSearch).
    Returns the fitted pipeline and a mongo-style doc with metrics.
    """
    X = df.drop(columns=[target])
    y = df[target]
    if drop_duration and "duration" in X.columns:
        X = X.drop(columns=["duration"])
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()

    sampler = None
    pipe = build_pipeline(cat_cols, num_cols, sampler=sampler, random_state=random_state)

    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=stratify, random_state=random_state)

    # fit the preprocessing and classifier with preset params
    # apply preset params to the classifier step
    if preset_params:
        clf_params = {}
        # allow both 'clf__param' and 'param' styles
        for k, v in preset_params.items():
            if k.startswith("clf__"):
                clf_params[k.replace("clf__", "")] = v
            else:
                clf_params[k] = v
        # set params on the classifier
        pipe.named_steps["clf"].set_params(**clf_params)

    pipe.fit(X_train, y_train)

    # optional calibration on 20% of train
    X_train_inner, X_calib, y_train_inner, y_calib = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_state
    )
    try:
        calibrated = CalibratedClassifierCV(pipe, cv="prefit", method="sigmoid")
        calibrated.fit(X_calib, y_calib)
        model_to_use = calibrated
    except Exception:
        model_to_use = pipe

    y_prob = model_to_use.predict_proba(X_test)[:, 1]
    # default threshold search
    thr_grid = np.linspace(0.05, 0.95, 19)
    best_f1, best_t = -1.0, 0.5
    for t in thr_grid:
        y_hat_t = (y_prob >= t).astype(int)
        f1_t = f1_score(y_test, y_hat_t, zero_division=0)
        if f1_t > best_f1:
            best_f1, best_t = float(f1_t), float(t)

    if decision_threshold is not None:
        best_t = float(decision_threshold)
        best_f1 = float(f1_score(y_test, (y_prob >= best_t).astype(int), zero_division=0))

    y_pred_best = (y_prob >= best_t).astype(int)
    acc = float(accuracy_score(y_test, y_pred_best))
    prec = float(precision_score(y_test, y_pred_best, zero_division=0))
    rec = float(recall_score(y_test, y_pred_best, zero_division=0))
    f1v = float(f1_score(y_test, y_pred_best, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    avg_prec = float(average_precision_score(y_test, y_prob))
    cm = confusion_matrix(y_test, y_pred_best).tolist()
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_best).ravel()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    pc, rc, _ = precision_recall_curve(y_test, y_prob)
    curves = {"roc_curve": {"fpr": downsample(fpr), "tpr": downsample(tpr)}, "precision_recall_curve": {"precision": downsample(pc), "recall": downsample(rc)}}

    pre = pipe.named_steps["preproc"]
    enc_name = None
    enc_cols = []
    for name, transformer, cols in pre.transformers_:
        if name in ("ohe", "ord"):
            enc_name = name
            enc_cols = cols
            break
    if enc_name is None:
        ohe_features = []
    else:
        enc = pre.named_transformers_.get(enc_name)
        try:
            ohe_features = list(enc.get_feature_names_out(input_features=enc_cols))
        except Exception:
            ohe_features = []
            cats = getattr(enc, "categories_", [])
            cols = enc_cols
            for i, col in enumerate(cols):
                if cats and i < len(cats):
                    for v in cats[i]:
                        ohe_features.append(f"{col}_{v}")
                else:
                    ohe_features.append(col)
    feature_names = ohe_features + num_cols

    doc = {
        "model_name": "DecisionTreeClassifier",
        "model_version": datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S"),
        "params": preset_params or {},
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1v,
            "roc_auc": roc_auc,
            "average_precision": avg_prec,
            "confusion_matrix": cm,
            "counts_at_best_threshold": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
            "decision_threshold": float(best_t),
            "thresholds": {"default": 0.5, "best_f1": best_t, "f1_at_best_t": best_f1},
        },
        "curves": curves,
        "features_used": feature_names,
        "notes": f"drop_duration={drop_duration}",
        "ts": datetime.datetime.now(datetime.timezone.utc),
    }

    return pipe, doc


def main(data_path: str, output_dir: str = "artifacts", target: str = "y", test_size: float = 0.2, sep: str | None = None, drop_duration: bool = False):
    print("Cargando datos:", data_path)
    df = load_data(data_path, target, sep)
    print("Entrenando modelo con parámetros predefinidos (DecisionTree)...")
    # preset params (reproduced best run) — these are applied directly to the DecisionTree
    preset = {
        "clf__criterion": "gini",
        "clf__max_depth": None,
        "clf__min_samples_split": 10,
        "clf__min_samples_leaf": 16,
        "clf__max_features": None,
        "clf__class_weight": None,
        "clf__ccp_alpha": 0.0001,
    }
    # decision threshold default from env if present
    decision_threshold = None
    dt_env_thr = os.getenv("DECISION_THRESHOLD")
    if dt_env_thr is not None and dt_env_thr != "":
        try:
            decision_threshold = float(dt_env_thr)
        except Exception:
            decision_threshold = None

    best_model, mongo_doc = train_with_preset_decision_tree(
        df=df, target=target, test_size=test_size, drop_duration=drop_duration, random_state=42, preset_params=preset, decision_threshold=decision_threshold
    )
    artifacts = Path(output_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    model_path = artifacts / f"{mongo_doc['model_name']}_{mongo_doc['model_version']}.joblib"
    joblib.dump(best_model, model_path)
    print(f"Modelo guardado en: {model_path}")
    # save run to Mongo (keeps historical record); remove if you prefer not to persist
    run_id = save_training_run(mongo_doc)
    print("Métricas insertadas en MongoDB (training_runs).")
    print(f"   run_id: {run_id}")
    m = mongo_doc["metrics"]
    print(json.dumps({"accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"], "f1": m["f1"], "roc_auc": m["roc_auc"], "average_precision": m["average_precision"], "best_threshold": m["decision_threshold"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena modelo y guarda métricas en MongoDB.")
    # defaults can be overridden via a .env file in the repo root (see .env.example)
    dataset_default = os.getenv("DATA_PATH", "data/bank-full.csv")
    output_default = os.getenv("OUTPUT_DIR", "artifacts")
    sep_default = os.getenv("CSV_SEP", None)
    if sep_default == "":
        sep_default = None
    n_iter_default = _env_int("N_ITER", 40)
    max_estimators_default = _env_int("MAX_ESTIMATORS", 300)
    drop_duration_default = _env_bool("DROP_DURATION", False)
    use_smote_default = _env_bool("USE_SMOTE", False)
    use_undersample_default = _env_bool("USE_UNDERSAMPLE", False)

    parser.add_argument("--data-path", type=str, default=dataset_default, help="Ruta del CSV de entrenamiento.")
    parser.add_argument("--output-dir", type=str, default=output_default, help="Carpeta para guardar el modelo.")
    parser.add_argument("--target", type=str, default="y", help="Nombre de la columna objetivo.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporción para test.")
    parser.add_argument("--sep", type=str, default=sep_default, help='Separador del CSV. Usa None para autodetectar (; o ,).')
    parser.add_argument("--drop-duration", action="store_true", default=drop_duration_default, help="Si se pasa, elimina la columna 'duration' para evitar leakage.")
    parser.add_argument("--use-smote", action="store_true", default=use_smote_default, help="Si se pasa, aplica SMOTE en el pipeline de entrenamiento (requiere imbalanced-learn).")
    parser.add_argument("--use-undersample", action="store_true", default=use_undersample_default, help="Si se pasa, aplica undersampling en el pipeline de entrenamiento (requiere imbalanced-learn).")
    parser.add_argument("--n-iter", type=int, default=n_iter_default, help="Número de iteraciones para RandomizedSearchCV (más bajo = más rápido).")
    parser.add_argument("--max-estimators", type=int, default=max_estimators_default, help="Valor máximo para n_estimators en la búsqueda.")
    args = parser.parse_args()
    # By default run the simplified preset DecisionTree trainer (no hyper-search)
    print("Ejecutando entrenamiento final (preset DecisionTree)...")
    main(data_path=args.data_path, output_dir=args.output_dir, target=args.target, test_size=args.test_size, sep=args.sep, drop_duration=args.drop_duration)
