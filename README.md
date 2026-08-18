# ML Assignment 2 — Bank Marketing Classification

## a. Problem Statement

This project predicts whether a bank client will subscribe to a term deposit after a direct-marketing contact. Five supervised classification algorithms are trained on the same UCI Bank Marketing dataset and evaluated on one reproducible 20% hold-out set. The deployed Streamlit application accepts labelled CSV test data, allows model selection, calculates the required metrics, and displays a confusion matrix and classification report.

## b. Dataset Description

- **Dataset:** Bank Marketing
- **Source:** UCI Machine Learning Repository, dataset ID 222
- **Problem type:** Binary classification
- **Instances used:** 45211
- **Input features:** 16
- **Target:** `target` where `0 = no subscription` and `1 = subscribed`
- **Split:** Stratified 80% train / 20% test
- **Random state:** 42

The dataset describes direct marketing campaigns of a Portuguese banking institution. Client attributes, loan information, contact details and campaign history are used to predict term-deposit subscription.

**Numerical features:** age, balance, day_of_week, duration, campaign, pdays, previous

**Categorical features:** job, marital, education, default, housing, loan, contact, month, poutcome

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

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9012 | 0.9056 | 0.6445 | 0.3478 | 0.4518 | 0.4261 |
| Decision Tree | 0.9008 | 0.8626 | 0.6182 | 0.3979 | 0.4842 | 0.4450 |
| kNN | 0.9017 | 0.8713 | 0.6634 | 0.3242 | 0.4356 | 0.4187 |
| Naive Bayes | 0.8548 | 0.8101 | 0.4059 | 0.5198 | 0.4559 | 0.3774 |
| Random Forest (Ensemble) | 0.9015 | 0.9261 | 0.7103 | 0.2665 | 0.3876 | 0.3956 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Provides a strong linear baseline with Accuracy=0.9012, AUC=0.9056 and F1=0.4518. The gap between precision (0.6445) and recall (0.3478) indicates how well the linear decision boundary handles the minority subscription class. |
| Decision Tree | Captures non-linear interactions between client and campaign attributes. It achieves F1=0.4842 and MCC=0.4450; comparison with the Random Forest shows whether a single tree is more sensitive to the specific split. |
| kNN | Uses standardized numerical features together with one-hot encoded categories. Its Recall=0.3242 and F1=0.4356; performance reflects the difficulty of neighborhood-based learning after mixed-type feature encoding. |
| Naive Bayes | Acts as a probabilistic baseline. AUC=0.8101, Precision=0.4059 and Recall=0.5198. Its conditional-independence assumption can be limiting because several marketing and client attributes are related. |
| Random Forest (Ensemble) | Combines many decision trees and uses class weighting for the imbalanced target. It obtains Accuracy=0.9015, AUC=0.9261, F1=0.3876 and MCC=0.3956, providing a more stable non-linear model than one tree. |
| Overall Winner | Decision Tree has the highest F1 score (0.4842) on the fixed test split. Highest AUC: Random Forest (Ensemble); highest Recall: Naive Bayes; highest MCC: Decision Tree. Because subscription is the minority outcome, F1 and MCC are considered together with Accuracy and AUC rather than choosing a model from Accuracy alone. |

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

- Random seed: 42
- Split: stratified 80% train / 20% test
- Positive class: term-deposit subscription (`yes` → `1`)
- Preprocessing is stored inside each scikit-learn pipeline.
- Model artifacts are serialized with joblib.
- The app checks incoming feature columns before prediction.
