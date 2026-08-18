from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

st.set_page_config(
    page_title="Breast Cancer Classification Lab",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 Breast Cancer Classification Lab")
st.caption(
    "Compare multiple classification models on the Wisconsin Diagnostic Breast Cancer dataset. "
    "Upload labelled test data, choose a model, and inspect predictions and evaluation metrics."
)


@st.cache_resource
def load_assets():
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    models = {
        name: joblib.load(MODEL_DIR / filename)
        for name, filename in metadata["model_files"].items()
    }
    reference_metrics = pd.read_csv(BASE_DIR / "metrics.csv", index_col=0)
    return metadata, models, reference_metrics


metadata, models, reference_metrics = load_assets()
expected_features = metadata["feature_names"]
target_col = metadata["target_column"]

with st.sidebar:
    st.header("Controls")
    selected_model_name = st.selectbox("Select classification model", list(models.keys()))
    st.markdown("---")
    st.write(f"**Dataset:** {metadata['dataset_name']}")
    st.write(f"**Instances:** {metadata['n_instances']}")
    st.write(f"**Features:** {metadata['n_features']}")
    st.write("**Target:** 0 = malignant, 1 = benign")

st.subheader("1. Upload test data")
uploaded_file = st.file_uploader(
    "Upload a CSV containing the 30 feature columns and, for evaluation, a `target` column.",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)
        source_label = "Uploaded CSV"
    except Exception as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")
        st.stop()
else:
    input_df = pd.read_csv(BASE_DIR / "test_data.csv")
    source_label = "Bundled test_data.csv (used because no file was uploaded)"
    st.info("No file uploaded yet. The app is demonstrating results using the bundled `test_data.csv`.")

st.write(f"**Current data:** {source_label} — {input_df.shape[0]} rows × {input_df.shape[1]} columns")
st.dataframe(input_df.head(10), use_container_width=True)

missing_features = [c for c in expected_features if c not in input_df.columns]
if missing_features:
    st.error(
        "The CSV is missing required model features: " + ", ".join(missing_features)
    )
    st.stop()

# Ignore unrelated columns and force training-time feature order.
X = input_df[expected_features].copy()
if X.isnull().any().any():
    st.error("The selected feature columns contain missing values. Please clean/impute the CSV before evaluation.")
    st.stop()

model = models[selected_model_name]
try:
    y_pred = model.predict(X)
    y_score = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)
except Exception as exc:
    st.error(f"Prediction failed: {exc}")
    st.stop()

st.subheader("2. Selected model")
st.success(f"Model: **{selected_model_name}**")

prediction_df = input_df.copy()
prediction_df["predicted_target"] = y_pred
prediction_df["predicted_class"] = prediction_df["predicted_target"].map({0: "malignant", 1: "benign"})
prediction_df["benign_probability_or_score"] = y_score

if target_col in input_df.columns:
    y_true = input_df[target_col]
    valid_targets = set(pd.Series(y_true).dropna().unique()).issubset({0, 1})
    if not valid_targets:
        st.error("`target` must contain only 0 and 1 for this binary classification dataset.")
        st.stop()

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score) if pd.Series(y_true).nunique() == 2 else float("nan"),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

    st.subheader("3. Evaluation metrics")
    cols = st.columns(6)
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label, "N/A" if pd.isna(value) else f"{value:.4f}")

    st.subheader("4. Confusion matrix and classification report")
    left, right = st.columns(2)
    with left:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Malignant (0)", "Benign (1)"],
        ).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"Confusion Matrix — {selected_model_name}")
        st.pyplot(fig, clear_figure=True)
    with right:
        report = classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Malignant", "Benign"],
            output_dict=True,
            zero_division=0,
        )
        st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)
else:
    st.warning(
        "No `target` column was found. Predictions are shown, but evaluation metrics and the confusion matrix require labelled test data."
    )

st.subheader("5. Predictions")
st.dataframe(
    prediction_df[["predicted_target", "predicted_class", "benign_probability_or_score"]].head(50),
    use_container_width=True,
)
st.download_button(
    "Download predictions as CSV",
    data=prediction_df.to_csv(index=False).encode("utf-8"),
    file_name="predictions.csv",
    mime="text/csv",
)

st.subheader("6. Reference comparison on the fixed 20% test split")
st.caption("These values were generated during model training using random_state=42 and stratified splitting.")
st.dataframe(reference_metrics.round(4), use_container_width=True)

winner = reference_metrics["F1"].idxmax()
st.write(f"**Highest F1 score on the reference split:** {winner} ({reference_metrics.loc[winner, 'F1']:.4f})")
