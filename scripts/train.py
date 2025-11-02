#!/usr/bin/env python3
"""Entrena un único árbol de decisión y guarda un pipeline calibrado (enfocado en AP)."""
from pathlib import Path
import argparse
import json
import datetime
import os
import sys

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Asegura que la raíz del proyecto esté en sys.path.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from integrations.featurize import featurize_df
from integrations.mongo_repo import save_training_run



def build_pipeline(cat_cols):
    """Pipeline: OneHot + DecisionTree (único)."""
    # Agrupa categorías infrecuentes para reducir ruido
    encoder = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100, sparse_output=False)
    preprocessor = ColumnTransformer([("onehot", encoder, cat_cols)], remainder="passthrough")

    classifier = DecisionTreeClassifier(
        random_state=42,
        class_weight=None,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        ccp_alpha=0.0,
    )

    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def load_dataset(path: str, sep: str) -> pd.DataFrame:
    delimiter = sep
    if delimiter in ("", "None", None):
        delimiter = None
    engine = "python" if delimiter is None else "c"
    df = pd.read_csv(path, sep=delimiter, engine=engine)
    if "y" not in df.columns:
        raise SystemExit("CSV must contain 'y' column as target")
    target = df["y"]
    if target.dtype == object:
        target = target.map({"yes": 1, "no": 0}).fillna(target)
    df["y"] = target.astype(int)
    return df


def train(data_path: str, output_dir: str, test_size: float, random_state: int, sep: str, no_feat: bool,
          optimize: str = "f1", beta: float = 1.0,
          min_precision: float | None = None, cost_fp: float | None = None, cost_fn: float | None = None,
          drop_duration: bool = False, tune_dt: bool = False, tune_scoring: str = "f1", tune_iter: int = 30, tune_folds: int = 3, pos_weight: float | None = None,
          calibration: str = "isotonic", resample: str = "over", resample_ratio: float = 0.5):
    load_dotenv()
    data = load_dataset(data_path, sep)

    features = data.drop(columns=["y"])
    if drop_duration:
        # Evita leakage: duración se conoce post-llamada
        for col in ["duration", "duration_bin", "duration_log"]:
            if col in features.columns:
                features = features.drop(columns=[col])
    if not no_feat:
        features = featurize_df(features)
    target = data["y"].astype(int)

    stratify = target if target.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, stratify=stratify, random_state=random_state
    )

    categorical_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    cat_indices = [X_train.columns.get_loc(c) for c in categorical_cols]
    pipeline = build_pipeline(categorical_cols)

    # Tuning opcional de hiperparámetros (árbol único)
    if tune_dt:
        param_dist = {
            "classifier__criterion": ["gini", "entropy", "log_loss"],
            "classifier__max_depth": [None] + list(range(4, 41)),
            "classifier__min_samples_split": list(range(2, 101, 2)),
            "classifier__min_samples_leaf": list(range(1, 51)),
            "classifier__max_features": [None, "sqrt", "log2"],
            "classifier__ccp_alpha": np.linspace(0.0, 0.01, 21),
            "classifier__class_weight": [None, "balanced"],
        }
        cv = StratifiedKFold(n_splits=int(tune_folds), shuffle=True, random_state=random_state)
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=int(tune_iter),
            scoring=tune_scoring,
            n_jobs=-1,
            cv=cv,
            random_state=random_state,
            verbose=0,
        )
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_


    X_model, X_cal, y_model, y_cal = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_state
    )
    # Rebalanceo opcional solo en el split de entrenamiento (no en calibración/test)
    if resample in ("over", "under", "smote"):
        strategy = float(resample_ratio)
        if resample == "over":
            sampler = RandomOverSampler(sampling_strategy=strategy, random_state=random_state)
            X_model, y_model = sampler.fit_resample(X_model, y_model)
        elif resample == "under":
            sampler = RandomUnderSampler(sampling_strategy=strategy, random_state=random_state)
            X_model, y_model = sampler.fit_resample(X_model, y_model)
        else:  # smote -> fallback to safe random oversampling due to categorical features
            sampler = RandomOverSampler(sampling_strategy=strategy, random_state=random_state)
            X_model, y_model = sampler.fit_resample(X_model, y_model)
    sample_weight = None
    if pos_weight is not None and resample in (None, "none"):
        # Usa sample_weight solo si no se remuestrea
        sample_weight = np.where(y_model.values == 1, float(pos_weight), 1.0)
    pipeline.fit(X_model, y_model, **({"classifier__sample_weight": sample_weight} if sample_weight is not None else {}))

    # Calibra probabilidades en un hold-out
    fitted_model = pipeline
    try:
        method = (calibration or "sigmoid").lower()
        if method in ("sigmoid", "isotonic"):
            calibrator = CalibratedClassifierCV(pipeline, cv="prefit", method=method)
            calibrator.fit(X_cal, y_cal)
            fitted_model = calibrator
        else:
            fitted_model = pipeline
    except Exception:
        pass

    y_prob = fitted_model.predict_proba(X_test)[:, 1]
    # Curvas ROC y PR (downsample para guardar en Mongo sin saturar)
    def _downsample(arr, n=200):
        arr = np.asarray(arr)
        if arr.size <= n:
            return arr.tolist()
        idx = np.linspace(0, arr.size - 1, n).astype(int)
        return arr[idx].tolist()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_prob)
    curves = {
        "roc_curve": {"fpr": _downsample(fpr), "tpr": _downsample(tpr)},
        "precision_recall_curve": {"precision": _downsample(pr_precision), "recall": _downsample(pr_recall)},
    }
    # Búsqueda del mejor umbral
    best_threshold = 0.5
    best_score = -1.0
    precision_target_met = True

    # Usa probabilidades únicas como candidatos
    uniq = np.unique(y_prob)
    candidate_thresholds = np.concatenate(([0.0], uniq, [1.0]))

    # Si hay piso de precisión, maximiza recall sujeto a precisión mínima
    if min_precision is not None and optimize == "recall":
        min_p = float(min_precision)
        best_rec = -1.0
        best_prec = -1.0
        chosen = None
        for threshold in candidate_thresholds:
            preds = (y_prob >= threshold).astype(int)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            if prec >= min_p:
                # maximize recall, tie-break by higher precision, then by lower threshold
                if rec > best_rec or (rec == best_rec and (prec > best_prec or (prec == best_prec and threshold < (chosen if chosen is not None else 1.0)))):
                    best_rec = rec
                    best_prec = prec
                    chosen = float(threshold)
        if chosen is None:
            # Si no se cumple, elige el umbral con mayor precisión
            precision_target_met = False
            best_prec = -1.0
            for threshold in candidate_thresholds:
                preds = (y_prob >= threshold).astype(int)
                prec = precision_score(y_test, preds, zero_division=0)
                if prec > best_prec:
                    best_prec = prec
                    chosen = float(threshold)
        best_threshold = float(chosen)
        y_pred = (y_prob >= best_threshold).astype(int)
    else:
    # Objetivo genérico (con filtro de piso de precisión si aplica)
        for threshold in candidate_thresholds:
            preds = (y_prob >= threshold).astype(int)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            if min_precision is not None and prec < float(min_precision):
                continue
            if optimize == "precision":
                score = prec
            elif optimize == "recall":
                score = rec
            elif optimize == "fbeta":
                b2 = float(beta) ** 2
                denom = b2 * prec + rec
                score = (1 + b2) * prec * rec / denom if denom > 0 else 0
            elif optimize == "cost" and cost_fp is not None and cost_fn is not None:
                from sklearn.metrics import confusion_matrix
                tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
                score = -(float(cost_fp) * fp + float(cost_fn) * fn)  # minimize cost -> maximize negative cost
            else:  # default F1
                score = f1_score(y_test, preds, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        y_pred = (y_prob >= best_threshold).astype(int)
    
    # Matriz de confusión y métricas derivadas
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "specificity": float(specificity),  # How well we identify negatives
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "average_precision": float(average_precision_score(y_test, y_prob)),
        "decision_threshold": best_threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "negative_predictive_value": float(tn / (tn + fn)) if (tn + fn) > 0 else 0,
        "precision_target": float(min_precision) if min_precision is not None else None,
        "precision_target_met": bool(precision_target_met),
    }

    # Empaqueta featurización + modelo para aceptar datos crudos
    packaged_model = fitted_model
    if not no_feat:
        packaged_model = Pipeline([
            ("featurize", FunctionTransformer(featurize_df, validate=False)),
            ("model", fitted_model),
        ])

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    # Nombre del artefacto con timestamp
    model_label = "DecisionTree"
    model_name = f"{model_label}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    artifact_path = output_path / f"{model_name}.joblib"
    joblib.dump(packaged_model, artifact_path)

    record = {
        "model_name": model_label,
        "model_version": model_name,
        "params": {"class_weight": "balanced"},
        "metrics": metrics,
        "curves": curves,
        "ts": datetime.datetime.now(datetime.timezone.utc),
    }
    run_id = save_training_run(record)

    print("Model saved:", artifact_path)
    print("run_id:", run_id)
    print(json.dumps(metrics, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Train a single DecisionTree model on the bank marketing dataset")
    parser.add_argument("--data-path", default=os.getenv("DATA_PATH", "data/bank-full.csv"))
    parser.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "artifacts"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=int(os.getenv("RANDOM_STATE", 42)))
    parser.add_argument("--sep", default=os.getenv("CSV_SEP", ";"))
    parser.add_argument("--no-feat", action="store_true", help="Disable internal featurization before training")
    parser.add_argument("--optimize", choices=["f1", "fbeta", "precision", "recall", "cost"], default=os.getenv("OPTIMIZE", "f1"))
    parser.add_argument("--beta", type=float, default=float(os.getenv("BETA", 1.0)), help="Beta for F-beta optimization")
    parser.add_argument("--min-precision", type=float, default=None, help="Require at least this precision when selecting threshold (omit by default)")
    parser.add_argument("--cost-fp", type=float, default=os.getenv("COST_FP", None), help="Cost of a false positive (used with --optimize cost)")
    parser.add_argument("--cost-fn", type=float, default=os.getenv("COST_FN", None), help="Cost of a false negative (used with --optimize cost)")
    parser.add_argument("--drop-duration", action="store_true", help="Drop 'duration' to avoid leakage")
    parser.add_argument("--tune-dt", action="store_true", default=True, help="Enable hyperparameter tuning for DecisionTree (RandomizedSearchCV)")
    parser.add_argument("--tune-scoring", choices=["f1", "roc_auc", "average_precision"], default=os.getenv("TUNE_SCORING", "average_precision"), help="Metric to optimize during hyperparameter search")
    parser.add_argument("--tune-iter", type=int, default=int(os.getenv("TUNE_ITER", 80)), help="Number of parameter settings sampled in RandomizedSearchCV")
    parser.add_argument("--tune-folds", type=int, default=int(os.getenv("TUNE_FOLDS", 5)), help="Number of CV folds for tuning")
    parser.add_argument("--pos-weight", type=float, default=os.getenv("POS_WEIGHT", None), help="Positive class weight for training (sample_weight)")
    parser.add_argument("--calibration", choices=["sigmoid", "isotonic", "none"], default=os.getenv("CALIBRATION", "sigmoid"))
    parser.add_argument("--resample", choices=["none", "over", "under", "smote"], default=os.getenv("RESAMPLE", "none"), help="Class balancing strategy for training split")
    parser.add_argument("--resample-ratio", type=float, default=float(os.getenv("RESAMPLE_RATIO", 0.5)), help="Sampling strategy ratio (minority/majority for over/SMOTE; target minority fraction for under)")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train(
        data_path=arguments.data_path,
        output_dir=arguments.output_dir,
        test_size=arguments.test_size,
        random_state=arguments.random_state,
        sep=arguments.sep,
        no_feat=arguments.no_feat,
        optimize=arguments.optimize,
        beta=arguments.beta,
        min_precision=arguments.min_precision,
        cost_fp=arguments.cost_fp,
        cost_fn=arguments.cost_fn,
        drop_duration=arguments.drop_duration,
        tune_dt=arguments.tune_dt,
        tune_scoring=arguments.tune_scoring,
        tune_iter=arguments.tune_iter,
        tune_folds=arguments.tune_folds,
        pos_weight=arguments.pos_weight,
        calibration=(None if arguments.calibration == "none" else arguments.calibration),
        resample=arguments.resample,
        resample_ratio=arguments.resample_ratio,
    )
