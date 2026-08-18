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
    page_title="Bank Term Deposit Classifier",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 Bank Term Deposit Subscription Classifier")
st.caption(
    "Compare five classification models trained on the UCI Bank Marketing dataset. "
    "Upload labelled test data, select a model, and inspect evaluation results and predictions."
)


@st.cache_resource
def load_project_assets():
    required = [MODEL_DIR / "metadata.json", BASE_DIR / "metrics.csv", BASE_DIR / "test_data.csv"]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing generated files: " + ", ".join(missing) + ". Run `python model/train_models.py` first."
        )

    metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    models = {}
    for display_name, filename in metadata["model_files"].items():
        path = MODEL_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing model artifact: {path.name}")
        models[display_name] = joblib.load(path)

    reference_metrics = pd.read_csv(BASE_DIR / "metrics.csv", index_col=0)
    return metadata, models, reference_metrics


try:
    metadata, models, reference_metrics = load_project_assets()
except Exception as exc:
    st.error(str(exc))
    st.info("Generate the artifacts on BITS Virtual Lab, commit them to GitHub, and redeploy the app.")
    st.stop()

expected_features = metadata["feature_names"]
target_col = metadata["target_column"]

with st.sidebar:
    st.header("Experiment controls")
    selected_model_name = st.selectbox(
        "Classification model",
        list(models.keys()),
    )
    st.divider()
    st.write(f"**Dataset:** {metadata['dataset_name']}")
    st.write(f"**Instances:** {metadata['n_instances']:,}")
    st.write(f"**Input features:** {metadata['n_features']}")
    st.write("**Positive class:** Subscribed (1)")
    st.write(f"**Random state:** {metadata['random_state']}")

st.subheader("1. Test-data input")
uploaded_file = st.file_uploader(
    "Upload a CSV with the 16 Bank Marketing feature columns. Include `target` (0/1) to calculate evaluation metrics.",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)
        source_label = "Uploaded CSV"
    except Exception as exc:
        st.error(f"The uploaded CSV could not be read: {exc}")
        st.stop()
else:
    input_df = pd.read_csv(BASE_DIR / "test_data.csv")
    source_label = "Bundled test_data.csv"
    st.info("No CSV uploaded. The app is using the committed 20% hold-out test set.")

st.write(f"**Current data:** {source_label} — {input_df.shape[0]:,} rows × {input_df.shape[1]} columns")
st.dataframe(input_df.head(10), use_container_width=True)

missing_features = [feature for feature in expected_features if feature not in input_df.columns]
if missing_features:
    st.error("Missing required feature columns: " + ", ".join(missing_features))
    st.stop()

X = input_df[expected_features].copy()
model = models[selected_model_name]

try:
    y_pred = model.predict(X)
    y_probability = model.predict_proba(X)[:, 1]
except Exception as exc:
    st.error(f"Prediction failed. Check the CSV values and columns. Details: {exc}")
    st.stop()

st.subheader("2. Selected model")
st.success(f"Evaluating **{selected_model_name}**")

prediction_df = input_df.copy()
prediction_df["predicted_target"] = y_pred.astype(int)
prediction_df["predicted_subscription"] = prediction_df["predicted_target"].map(
    {0: "No", 1: "Yes"}
)
prediction_df["subscription_probability"] = y_probability

if target_col in input_df.columns:
    y_true = pd.to_numeric(input_df[target_col], errors="coerce")
    if y_true.isna().any() or not set(y_true.unique()).issubset({0, 1}):
        st.error("`target` must contain only 0 and 1.")
        st.stop()
    y_true = y_true.astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_probability) if y_true.nunique() == 2 else float("nan"),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

    st.subheader("3. Evaluation metrics")
    metric_columns = st.columns(6)
    for container, (name, value) in zip(metric_columns, metrics.items()):
        container.metric(name, "N/A" if pd.isna(value) else f"{value:.4f}")

    st.subheader("4. Confusion matrix and classification report")
    left, right = st.columns(2)
    with left:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No subscription", "Subscribed"],
        ).plot(ax=ax, colorbar=False)
        ax.set_title(f"Confusion Matrix — {selected_model_name}")
        st.pyplot(fig, clear_figure=True)
    with right:
        report = classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["No subscription", "Subscribed"],
            output_dict=True,
            zero_division=0,
        )
        st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)
else:
    st.warning(
        "No `target` column was found. Predictions are available, but labelled test data is required for evaluation metrics."
    )

st.subheader("5. Predictions")
st.dataframe(
    prediction_df[["predicted_target", "predicted_subscription", "subscription_probability"]].head(100),
    use_container_width=True,
)
st.download_button(
    "Download predictions",
    data=prediction_df.to_csv(index=False).encode("utf-8"),
    file_name="bank_marketing_predictions.csv",
    mime="text/csv",
)

st.subheader("6. Model comparison on the fixed 20% test split")
st.caption(
    "Reference values generated by model/train_models.py using stratified splitting and random_state=42."
)
st.dataframe(reference_metrics.round(4), use_container_width=True)

winner = reference_metrics["F1"].idxmax()
st.write(
    f"**Highest F1 on the reference split:** {winner} "
    f"({reference_metrics.loc[winner, 'F1']:.4f})"
)
