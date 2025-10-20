import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    classification_report,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text


def main(data_path, output_dir, target, test_size):
    # =======================
    # CARGA DE DATOS
    # =======================
    df = pd.read_csv(data_path, sep=";")
    df[target] = df[target].map({"yes": 1, "no": 0})
    X = df.drop(columns=[target])
    y = df[target]

    # =======================
    # PREPROCESAMIENTO
    # =======================
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        [("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)],
        remainder="passthrough",
    )

    pipe = Pipeline([("preproc", preprocessor), ("clf", DecisionTreeClassifier(random_state=42))])

    # =======================
    # DIVISIÓN TRAIN/TEST
    # =======================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y if len(y.unique()) > 1 else None, random_state=42
    )

    # =======================
    # GRID SEARCH
    # =======================
    param_grid = {
        "clf__criterion": ["gini", "entropy"],
        "clf__max_depth": [3, 5, 7, None],
        "clf__min_samples_split": [2, 10, 20],
        "clf__class_weight": [None, "balanced"],
    }

    gs = GridSearchCV(pipe, param_grid, scoring="f1", cv=5, n_jobs=-1)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_

    # =======================
    # PREDICCIÓN Y MÉTRICAS
    # =======================
    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, y_prob))
    cm = confusion_matrix(y_test, y_pred).tolist()
    cr_str = classification_report(y_test, y_pred)

    # =======================
    # CURVAS ROC Y PR
    # =======================
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)

    curves = {
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "precision_recall_curve": {"precision": prec_curve.tolist(), "recall": rec_curve.tolist()},
    }

    # =======================
    # GUARDAR MODELO Y MÉTRICAS
    # =======================
    artifacts = Path(output_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    model_path = artifacts / "decision_tree_model.joblib"
    joblib.dump(best, model_path)

    metrics = {
        "model": "DecisionTreeClassifier",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "classification_report": cr_str,
        "best_params": gs.best_params_,
    }

    with open(artifacts / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)

    with open(artifacts / "curves.json", "w", encoding="utf-8") as f:
        json.dump(curves, f, ensure_ascii=False, indent=2, default=str)

    # =======================
    # EXPORTAR REGLAS DEL ÁRBOL
    # =======================
    pre = best.named_steps["preproc"]
    ohe = pre.named_transformers_["ohe"]

    try:
        ohe_features = list(ohe.get_feature_names_out(cat_cols))
    except Exception:
        ohe_features = []
        for i, col in enumerate(cat_cols):
            for v in ohe.categories_[i]:
                ohe_features.append(f"{col}_{v}")

    feature_names = ohe_features + num_cols
    dt = best.named_steps["clf"]
    rules = export_text(dt, feature_names=feature_names)

    print(f"✅ Modelo guardado en: {model_path}")
    print(f"✅ Métricas y curvas guardadas en: {artifacts}")
    print("\n==== Reglas del Árbol de Decisión ====\n")
    print(rules)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/bank-full.csv")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--target", type=str, default="y")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    main(args.data_path, args.output_dir, args.target, args.test_size)
