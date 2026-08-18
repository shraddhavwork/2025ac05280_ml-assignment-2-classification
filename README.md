# Machine Learning Assignment 2 — Classification & Streamlit Deployment

## a. Problem Statement

This project implements and compares multiple supervised machine-learning classification models on a single public dataset and exposes the trained models through an interactive Streamlit application. The application accepts test data in CSV format, lets the user select a classification model, displays the required evaluation metrics, and shows a confusion matrix and classification report.

The assignment PDF contains a wording inconsistency: it says “all 6 ML models” in one sentence, but the numbered model list and the required comparison table contain **five** classifiers. This project follows the five explicitly named models rather than inventing an unspecified sixth algorithm.

## b. Dataset Description

**Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC)  
**Original source:** UCI Machine Learning Repository  
**Problem type:** Binary classification  
**Instances:** 569  
**Input features:** 30 numeric features  
**Target:** `target` where `0 = malignant` and `1 = benign`

The 30 predictors describe characteristics computed from digitized images of breast-mass cell nuclei, including radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension. The dataset satisfies the assignment requirement of at least 500 instances and 12 features.

The experiment uses a **stratified 80/20 train-test split** with `random_state=42`. The resulting held-out test set contains 114 rows and is included as `test_data.csv` for the Streamlit upload/evaluation workflow.

## c. GitHub Repository Link

**GitHub Repository:** `https://github.com/shraddhavwork/2025ac05280_ml-assignment-2-classification`

**Live Streamlit App:** `hhttps://2025ac05280ml-assignment-2-classification-wpbt7pwrvhiuzvxqchi5.streamlit.app/`

> Replace both placeholders after you push the repository and deploy the application.

## d. Models Used and Evaluation Metrics

The project evaluates the following classifiers:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)

The required evaluation metrics are **Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC)**.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9035 | 0.9377 | 0.9692 | 0.8750 | 0.9197 | 0.8062 |
| K-Nearest Neighbors | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest (Ensemble) | 0.9474 | 0.9937 | 0.9583 | 0.9583 | 0.9583 | 0.8869 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Excellent overall performance, with 98.25% accuracy, very high AUC, and balanced precision/recall. It is also easy to interpret and computationally efficient. |
| Decision Tree | Lowest performance among the evaluated models on this split. Its recall and MCC are lower, indicating more classification errors and weaker overall agreement than the other models. |
| K-Nearest Neighbors | Very strong performance after feature scaling. It achieved perfect recall (1.0000), meaning all benign cases (positive class in this encoding) in the held-out test set were detected. |
| Naive Bayes | Good baseline performance with high AUC and recall, although its accuracy and MCC are below the best-performing models. Its conditional-independence assumption is relatively strong for these correlated medical features. |
| Random Forest (Ensemble) | Strong and stable ensemble performance with high AUC, precision, recall, F1, and MCC. It performs better than the single Decision Tree and is less sensitive to one tree's variance. |
| **Overall Winner for this dataset** | **Logistic Regression is the overall winner on this reproducible test split because it has the highest Accuracy (0.9825), F1 (0.9861), MCC (0.9623), and AUC (0.9954) among the five required models.** |

## Streamlit Application Features

The app includes all required assignment features:

- CSV test-data upload option
- Model-selection dropdown
- Accuracy, AUC, Precision, Recall, F1, and MCC display
- Confusion matrix
- Classification report
- Model predictions table
- Downloadable prediction CSV
- Reference comparison table for all trained models
- A built-in fallback to `test_data.csv` so the deployed app demonstrates results immediately even before a file is uploaded

## Repository Structure

```text
ML_Assignment_2_Project/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── breast_cancer_wisconsin.csv
├── metrics.csv
├── .gitignore
└── model/
    ├── model_training.ipynb
    ├── train_models.py
    ├── metadata.json
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── k_nearest_neighbors.joblib
    ├── naive_bayes.joblib
    ├── random_forest.joblib
```

## How to Run Locally / on BITS Virtual Lab

1. Open a terminal in the project folder.
2. Create/activate a Python environment if required by the lab.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the training notebook `model/model_training.ipynb` in Jupyter **or** run:

```bash
python model/train_models.py
```

5. Start Streamlit:

```bash
streamlit run app.py
```

6. Open the displayed local URL in the browser.
7. Upload `test_data.csv`, select different models, and verify the metrics/confusion matrix.
8. Capture **one screenshot showing the assignment execution on the BITS Virtual Lab** for the final submission PDF.

## Reproducibility Notes

- Random seed: `42`
- Split: stratified 80% train / 20% test
- Preprocessing is stored inside scikit-learn pipelines for algorithms that require feature scaling.
- Model artifacts are saved with `joblib` and loaded by the Streamlit app.
- The app validates the uploaded feature columns before prediction.
