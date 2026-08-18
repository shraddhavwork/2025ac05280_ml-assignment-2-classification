"""Train and persist classification models for ML Assignment 2.

Dataset: Wisconsin Diagnostic Breast Cancer (UCI; available through scikit-learn).
This project implements the five classifiers explicitly listed in the assignment.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            max_depth=5,
            min_samples_leaf=2,
        ),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=7)),
        ]),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }


def score_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_score),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def main():
    data = load_breast_cancer(as_frame=True)
    X = data.data.copy()
    y = data.target.copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    test_df = X_test.copy()
    test_df["target"] = y_test.values
    test_df.to_csv(BASE_DIR / "test_data.csv", index=False)

    full_df = X.copy()
    full_df["target"] = y.values
    full_df.to_csv(BASE_DIR / "breast_cancer_wisconsin.csv", index=False)

    models = build_models()
    metrics = {}
    model_files = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics[name] = score_model(model, X_test, y_test)
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        file_name = f"{safe_name}.joblib"
        joblib.dump(model, MODEL_DIR / file_name)
        model_files[name] = file_name

    metrics_df = pd.DataFrame(metrics).T
    metrics_df.index.name = "ML Model Name"
    metrics_df.to_csv(BASE_DIR / "metrics.csv")

    metadata = {
        "dataset_name": "Wisconsin Diagnostic Breast Cancer",
        "dataset_source": "UCI Machine Learning Repository (also packaged in scikit-learn)",
        "n_instances": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "target_column": "target",
        "target_mapping": {"0": "malignant", "1": "benign"},
        "feature_names": list(X.columns),
        "random_state": RANDOM_STATE,
        "test_size": 0.20,
        "model_files": model_files,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(metrics_df.round(4))
    print(f"\nSaved {len(models)} models in: {MODEL_DIR}")


if __name__ == "__main__":
    main()
