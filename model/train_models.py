"""Train and save all classification models required for ML Assignment 2.

Dataset: UCI Bank Marketing (dataset id 222)
Task: Predict whether a client subscribes to a term deposit (yes/no).

Run from the project root:
    python model/train_models.py

The script downloads the public dataset through the official ``ucimlrepo``
client, creates a reproducible stratified 80/20 split, trains five models,
and writes the test CSV, model artifacts, metrics, metadata, and README results.
"""
from __future__ import annotations

from pathlib import Path
import json
import joblib
import pandas as pd

from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.20
UCI_DATASET_ID = 222

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_bank_marketing() -> tuple[pd.DataFrame, pd.Series]:
    """Fetch the official UCI Bank Marketing data and return X and binary y."""
    dataset = fetch_ucirepo(id=UCI_DATASET_ID)
    X = dataset.data.features.copy()
    raw_y = dataset.data.targets.copy()

    if isinstance(raw_y, pd.DataFrame):
        if raw_y.shape[1] != 1:
            raise ValueError(f"Expected one target column, found {raw_y.shape[1]}.")
        raw_y = raw_y.iloc[:, 0]

    y_text = raw_y.astype(str).str.strip().str.lower()
    y = y_text.map({"no": 0, "yes": 1})
    if y.isna().any():
        unexpected = sorted(y_text[y.isna()].unique().tolist())
        raise ValueError(f"Unexpected target values returned by UCI: {unexpected}")
    y = y.astype(int).rename("target")

    if X.shape[0] < 500 or X.shape[1] < 12:
        raise ValueError(
            f"Dataset does not satisfy assignment minimums: {X.shape[0]} rows, {X.shape[1]} features."
        )
    return X, y


def feature_groups(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numerical = [column for column in X.columns if column not in categorical]
    return numerical, categorical


def make_preprocessor(
    numerical: list[str],
    categorical: list[str],
    *,
    scale_numeric: bool,
) -> ColumnTransformer:
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_models(numerical: list[str], categorical: list[str]) -> dict[str, Pipeline]:
    """Create the five classifiers explicitly listed in the assignment."""
    scaled = lambda: make_preprocessor(numerical, categorical, scale_numeric=True)
    unscaled = lambda: make_preprocessor(numerical, categorical, scale_numeric=False)

    return {
        "Logistic Regression": Pipeline(
            [
                ("preprocess", scaled()),
                (
                    "classifier",
                    LogisticRegression(max_iter=2500, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("preprocess", unscaled()),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        random_state=RANDOM_STATE,
                        max_depth=8,
                        min_samples_leaf=4,
                    ),
                ),
            ]
        ),
        "kNN": Pipeline(
            [
                ("preprocess", scaled()),
                ("classifier", KNeighborsClassifier(n_neighbors=11)),
            ]
        ),
        "Naive Bayes": Pipeline(
            [
                ("preprocess", scaled()),
                ("classifier", GaussianNB()),
            ]
        ),
        "Random Forest (Ensemble)": Pipeline(
            [
                ("preprocess", unscaled()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_score),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def model_filename(model_name: str) -> str:
    replacements = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "kNN": "knn.joblib",
        "Naive Bayes": "naive_bayes.joblib",
        "Random Forest (Ensemble)": "random_forest.joblib",
    }
    return replacements[model_name]


def build_observations(metrics_df: pd.DataFrame) -> dict[str, str]:
    winner = metrics_df["F1"].idxmax()
    best_auc = metrics_df["AUC"].idxmax()
    best_recall = metrics_df["Recall"].idxmax()
    best_mcc = metrics_df["MCC"].idxmax()

    lr = metrics_df.loc["Logistic Regression"]
    dt = metrics_df.loc["Decision Tree"]
    knn = metrics_df.loc["kNN"]
    nb = metrics_df.loc["Naive Bayes"]
    rf = metrics_df.loc["Random Forest (Ensemble)"]

    observations = {
        "Logistic Regression": (
            f"Provides a strong linear baseline with Accuracy={lr['Accuracy']:.4f}, "
            f"AUC={lr['AUC']:.4f} and F1={lr['F1']:.4f}. The gap between precision "
            f"({lr['Precision']:.4f}) and recall ({lr['Recall']:.4f}) indicates how well "
            "the linear decision boundary handles the minority subscription class."
        ),
        "Decision Tree": (
            f"Captures non-linear interactions between client and campaign attributes. "
            f"It achieves F1={dt['F1']:.4f} and MCC={dt['MCC']:.4f}; comparison with the "
            "Random Forest shows whether a single tree is more sensitive to the specific split."
        ),
        "kNN": (
            f"Uses standardized numerical features together with one-hot encoded categories. "
            f"Its Recall={knn['Recall']:.4f} and F1={knn['F1']:.4f}; performance reflects the "
            "difficulty of neighborhood-based learning after mixed-type feature encoding."
        ),
        "Naive Bayes": (
            f"Acts as a probabilistic baseline. AUC={nb['AUC']:.4f}, Precision={nb['Precision']:.4f} "
            f"and Recall={nb['Recall']:.4f}. Its conditional-independence assumption can be limiting "
            "because several marketing and client attributes are related."
        ),
        "Random Forest (Ensemble)": (
            f"Combines many decision trees and uses class weighting for the imbalanced target. "
            f"It obtains Accuracy={rf['Accuracy']:.4f}, AUC={rf['AUC']:.4f}, F1={rf['F1']:.4f} "
            f"and MCC={rf['MCC']:.4f}, providing a more stable non-linear model than one tree."
        ),
        "Overall Winner": (
            f"{winner} has the highest F1 score ({metrics_df.loc[winner, 'F1']:.4f}) on the fixed "
            f"test split. Highest AUC: {best_auc}; highest Recall: {best_recall}; highest MCC: {best_mcc}. "
            "Because subscription is the minority outcome, F1 and MCC are considered together with "
            "Accuracy and AUC rather than choosing a model from Accuracy alone."
        ),
    }
    return observations


def render_readme(metrics_df: pd.DataFrame, metadata: dict, observations: dict[str, str]) -> str:
    display_df = metrics_df.reset_index().copy()
    for column in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
        display_df[column] = display_df[column].map(lambda value: f"{value:.4f}")
    headers = display_df.columns.tolist()
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in display_df.astype(str).itertuples(index=False, name=None):
        table_lines.append("| " + " | ".join(row) + " |")
    table = "\n".join(table_lines)
    obs_rows = "\n".join(
        f"| {name} | {text} |" for name, text in observations.items()
    )
    categorical = ", ".join(metadata["categorical_features"])
    numerical = ", ".join(metadata["numerical_features"])

    return f"""# ML Assignment 2 — Bank Marketing Classification

## a. Problem Statement

This project predicts whether a bank client will subscribe to a term deposit after a direct-marketing contact. Five supervised classification algorithms are trained on the same UCI Bank Marketing dataset and evaluated on one reproducible 20% hold-out set. The deployed Streamlit application accepts labelled CSV test data, allows model selection, calculates the required metrics, and displays a confusion matrix and classification report.

## b. Dataset Description

- **Dataset:** Bank Marketing
- **Source:** UCI Machine Learning Repository, dataset ID 222
- **Problem type:** Binary classification
- **Instances used:** {metadata['n_instances']}
- **Input features:** {metadata['n_features']}
- **Target:** `target` where `0 = no subscription` and `1 = subscribed`
- **Split:** Stratified 80% train / 20% test
- **Random state:** {metadata['random_state']}

The dataset describes direct marketing campaigns of a Portuguese banking institution. Client attributes, loan information, contact details and campaign history are used to predict term-deposit subscription.

**Numerical features:** {numerical}

**Categorical features:** {categorical}

## c. Submission Links

- **GitHub Repository:** REPLACE_WITH_YOUR_GITHUB_REPOSITORY_URL
- **Live Streamlit App:** REPLACE_WITH_YOUR_STREAMLIT_APP_URL

## d. Models Used and Evaluation Metrics

The following classifiers are implemented on the same train/test split:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)

For categorical fields, the pipelines use one-hot encoding. Numerical scaling is applied to Logistic Regression, kNN and Naive Bayes. The required metrics are Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC).

### Comparison Table

{table}

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
{obs_rows}

## Streamlit Application Features

- CSV test-data upload
- Model-selection dropdown
- Accuracy, AUC, Precision, Recall, F1 and MCC
- Confusion matrix
- Classification report
- Prediction table with subscription probability
- Downloadable prediction CSV
- Reference comparison table for all trained models
- Built-in fallback to the committed `test_data.csv`

## Repository Structure

```text
.
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── metrics.csv
├── .gitignore
└── model/
    ├── model_training.ipynb
    ├── train_models.py
    ├── metadata.json
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── random_forest.joblib
```

## How to Reproduce the Training

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the training pipeline:

```bash
python model/train_models.py
```

The script downloads UCI dataset 222 using `ucimlrepo`, performs the fixed split, trains all five models and regenerates the model files, `test_data.csv`, `metrics.csv`, `metadata.json`, and the metric/observation sections of this README.

3. Start the Streamlit application:

```bash
streamlit run app.py
```

4. Upload `test_data.csv` (or use the bundled fallback), select different models and verify the metrics and confusion matrix.

## Reproducibility Notes

- Random seed: {metadata['random_state']}
- Split: stratified {int((1-metadata['test_size'])*100)}% train / {int(metadata['test_size']*100)}% test
- Positive class: term-deposit subscription (`yes` → `1`)
- Preprocessing is stored inside each scikit-learn pipeline.
- Model artifacts are serialized with joblib.
- The app checks incoming feature columns before prediction.
"""


def main() -> None:
    print("Fetching UCI Bank Marketing dataset (id=222)...")
    X, y = load_bank_marketing()
    numerical, categorical = feature_groups(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    test_df = X_test.copy()
    test_df["target"] = y_test.to_numpy()
    test_df.to_csv(BASE_DIR / "test_data.csv", index=False)

    models = build_models(numerical, categorical)
    results: dict[str, dict[str, float]] = {}
    model_files: dict[str, str] = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        results[name] = evaluate(model, X_test, y_test)
        filename = model_filename(name)
        joblib.dump(model, MODEL_DIR / filename)
        model_files[name] = filename

    metrics_df = pd.DataFrame.from_dict(results, orient="index")
    metrics_df.index.name = "ML Model Name"
    metrics_df.to_csv(BASE_DIR / "metrics.csv")

    metadata = {
        "dataset_name": "UCI Bank Marketing",
        "dataset_id": UCI_DATASET_ID,
        "n_instances": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "target_column": "target",
        "target_mapping": {"0": "No subscription", "1": "Subscribed"},
        "feature_names": X.columns.tolist(),
        "numerical_features": numerical,
        "categorical_features": categorical,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "model_files": model_files,
    }
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    observations = build_observations(metrics_df)
    readme = render_readme(metrics_df, metadata, observations)
    (BASE_DIR / "README.md").write_text(readme, encoding="utf-8")

    print("\nModel comparison:\n")
    print(metrics_df.round(4))
    print(f"\nSaved test data: {BASE_DIR / 'test_data.csv'}")
    print(f"Saved metrics:   {BASE_DIR / 'metrics.csv'}")
    print(f"Saved models:    {MODEL_DIR}")
    print("README.md has been refreshed with the actual metrics and observations.")


if __name__ == "__main__":
    main()
