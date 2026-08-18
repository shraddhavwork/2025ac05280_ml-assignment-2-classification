# ML Assignment 2 — Bank Marketing Classification

> **Before the final GitHub upload:** run `python model/train_models.py` once on BITS Virtual Lab. The script fetches the official UCI Bank Marketing dataset, generates the 20% test CSV, trains all five models, saves their `.joblib` files, calculates all six required metrics, and automatically rewrites this README with the actual comparison table and observations.

## a. Problem Statement

Predict whether a bank client will subscribe to a term deposit after a direct-marketing contact using the UCI Bank Marketing dataset.

## b. Dataset Description

- UCI Machine Learning Repository — Bank Marketing (dataset ID 222)
- Binary classification
- 45,211 instances
- 16 input features
- Target: subscription (`yes` / `no`)

## c. Submission Links

- **GitHub Repository:** REPLACE_WITH_YOUR_GITHUB_REPOSITORY_URL
- **Live Streamlit App:** REPLACE_WITH_YOUR_STREAMLIT_APP_URL

## d. Models Used

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)

Required metrics: Accuracy, AUC, Precision, Recall, F1 and MCC.

## Generate the final repository files

```bash
pip install -r requirements.txt
python model/train_models.py
```

After the command finishes, this README will contain the actual metric table and observations.
