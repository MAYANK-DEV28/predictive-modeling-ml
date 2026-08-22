# Predictive Modeling with Machine Learning — Breast Cancer Diagnosis

A complete, beginner-friendly supervised machine learning project that predicts whether a breast tumor is **malignant** or **benign** from measurements taken from a biopsy image.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Dataset](#dataset)
4. [Technologies Used](#technologies-used)
5. [Machine Learning Workflow](#machine-learning-workflow)
6. [Exploratory Data Analysis](#exploratory-data-analysis)
7. [Models Used](#models-used)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Model Comparison](#model-comparison)
10. [Key Findings](#key-findings)
11. [Limitations](#limitations)
12. [Future Improvements](#future-improvements)
13. [How to Run the Project](#how-to-run-the-project)
14. [Project Structure](#project-structure)

## Project Overview

This project builds and evaluates several supervised machine learning classifiers that learn from historical, labeled biopsy measurements to predict a diagnosis on unseen data. It follows a full, standard ML workflow: dataset documentation, exploratory data analysis (EDA), preprocessing, model training, evaluation, and visualization, and ends with a comparison of models and a justified final model choice.

## Problem Statement

Given 30 numeric measurements describing the cell nuclei found in a breast mass biopsy image, predict whether the mass is **malignant** (cancerous) or **benign** (non-cancerous).

- **Task type:** Binary classification
- **Why it's meaningful:** Consistent, quantitative screening support can help flag likely-malignant cases for closer review. This is a well-known, clean, real-world medical dataset that is ideal for demonstrating the full supervised learning workflow end-to-end. (See [Limitations](#limitations) — this is a learning project, not a validated clinical tool.)

## Dataset

- **Name:** Breast Cancer Wisconsin (Diagnostic) Data Set
- **Source:** Bundled with scikit-learn (`sklearn.datasets.load_breast_cancer`), originally from the UCI Machine Learning Repository. Loading it this way keeps the project fully reproducible offline — no external download is required. See `data/README.md` for details.
- **Rows:** 569 samples
- **Columns:** 30 numeric input features + 1 target column
- **Feature descriptions:** For each cell nucleus, the *mean*, *standard error*, and *"worst" (largest)* value of 10 base measurements were computed: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension. This gives 10 × 3 = 30 features.
- **Target variable:** `diagnosis` — `malignant` (encoded 0) or `benign` (encoded 1)
- **Data types:** All 30 input features are `float64`; the target is an integer/categorical label
- **Missing values:** None found (0 across all columns)
- **Duplicate records:** None found
- **Class distribution:** 357 benign (62.7%) vs. 212 malignant (37.3%) — moderately imbalanced, which is why accuracy alone is not used as the sole evaluation metric

## Technologies Used

- Python 3
- Pandas, NumPy — data manipulation
- Matplotlib, Seaborn — visualization
- Scikit-learn — preprocessing, models, evaluation metrics

## Machine Learning Workflow

1. **Load data** from scikit-learn's built-in dataset loader
2. **Explore** the data (shape, types, missing values, duplicates, class balance, correlations, distributions)
3. **Preprocess:** split into train/test sets (stratified, `random_state=42`), then fit a `StandardScaler` on the training data only and apply it to both splits (no data leakage)
4. **Train** three classifiers: Logistic Regression, Decision Tree, Random Forest
5. **Evaluate** each model with Accuracy, Precision, Recall, F1-score, Confusion Matrix, and ROC-AUC
6. **Visualize** results (confusion matrix, ROC curves, feature importance)
7. **Compare** models on the same test set and select the best one, with justification

## Exploratory Data Analysis

The full EDA (sample records, feature statistics, missing-value/duplicate checks, correlation analysis, and distribution plots by class) is in [`notebooks/predictive_modeling.ipynb`](notebooks/predictive_modeling.ipynb). Highlights:

- No missing values and no duplicate rows — the dataset needed no cleaning.
- All 30 predictors are numeric; the only categorical variable is the target.
- Size-related features (radius, perimeter, area) are highly correlated with each other, as expected, since they all describe the same underlying cell size.
- Malignant tumors show visibly larger and more irregular measurements (e.g. `worst area`, `worst concave points`) than benign ones — a good sign that the classes are separable.

## Models Used

| Model | Why included |
|---|---|
| Logistic Regression | Simple, interpretable linear baseline |
| Decision Tree Classifier | Single interpretable tree (max depth 5 to limit overfitting) |
| Random Forest Classifier | Ensemble of 300 trees; typically more robust than a single tree |

All models were trained with `random_state=42` for reproducibility, on the same scaled 80/20 train/test split.

## Evaluation Metrics

| Metric | What it means | Why it's relevant here |
|---|---|---|
| **Accuracy** | Fraction of all predictions that were correct | Easy to interpret, but can be misleading with the ~63/37 class split |
| **Precision** | Of predicted-benign cases, how many were truly benign | Matters if false "all clear" calls are costly |
| **Recall** | Of truly malignant cases, how many were correctly caught | **Most important metric for this problem** — missing a malignant tumor (a false negative) is far more costly than a false alarm |
| **F1-score** | Harmonic mean of precision and recall | A single balanced summary of the two |
| **ROC-AUC** | Probability the model ranks a random malignant case above a random benign case, across all thresholds | Threshold-independent view of separability; **used as the primary metric to select the best model** |

Given the moderate class imbalance and the cost asymmetry of missing a malignant case, **ROC-AUC** was chosen as the primary selection metric, with recall on the malignant class checked directly via the confusion matrix.

## Model Comparison

Results on the held-out 20% test set (114 samples), computed directly from the trained models — see `results/model_comparison.csv` / `results/metrics.json`:

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| -------------------- | -------: | --------: | -----: | -------: | ------: |
| **Logistic Regression** | **0.9825** | **0.9861** | **0.9861** | **0.9861** | **0.9954** |
| Random Forest         |   0.9474 |    0.9583 |  0.9583 |    0.9583 |  0.9937 |
| Decision Tree          |   0.9211 |    0.9565 |  0.9167 |    0.9362 |  0.9163 |

*(Precision/Recall/F1 above are for the positive class as defined by scikit-learn, `benign`. Per-class malignant recall is discussed below and is visible directly in the confusion matrix.)*

![Model comparison — ROC curves](results/roc_curve.png)

## Key Findings

- **Logistic Regression was the best-performing model** on this test set, with the highest accuracy (98.2%) and ROC-AUC (0.995).
  - Its confusion matrix shows only **2 misclassifications out of 114** test samples: 1 malignant case predicted as benign, and 1 benign case predicted as malignant. On the malignant class specifically, recall was **0.98** (41 of 42 correctly identified).
- **Random Forest was a close second** (ROC-AUC 0.994), with a malignant-class recall of 0.93 — solid, but Logistic Regression separated the classes slightly better here.
- **Decision Tree performed noticeably worse** (ROC-AUC 0.916) than the other two, which is expected: a single depth-limited tree has less capacity than an ensemble or a well-regularized linear model on this relatively small, well-separated dataset.
- **Why Logistic Regression won:** With 30 numeric, standardized features and classes that are largely linearly separable (as seen in the EDA distribution plots), a well-regularized linear model can draw an effective decision boundary without the variance that a single tree is prone to. This is a good reminder that more complex models (trees, forests) aren't automatically better — it depends on the structure of the data.
- **Feature importance (from the Random Forest)** shows that `worst perimeter`, `worst area`, `worst concave points`, `mean concave points`, and `worst radius` were the most influential predictors — consistent with the EDA finding that malignant tumors tend to be larger and more irregularly shaped.

![Feature importance](results/feature_importance.png)

![Confusion matrix — best model](results/confusion_matrix.png)

## Limitations

- **Dataset size and source:** 569 samples from a single, well-curated public dataset. Real-world clinical data is typically noisier, more varied across imaging equipment/labs, and larger; performance here is unlikely to generalize as-is to a different patient population or imaging pipeline.
- **No hyperparameter tuning:** Models were trained with reasonable default/lightly-constrained settings, not a full search (e.g. `GridSearchCV`), so reported performance is a baseline, not an upper bound.
- **Single train/test split:** Results come from one 80/20 split; a more rigorous evaluation would use k-fold cross-validation to check how stable these metrics are.
- **No external/held-out validation set** from a different source or time period.
- **This model is a learning/demonstration project, not a validated diagnostic tool**, and must not be used for real medical decision-making.

## Future Improvements

- Add k-fold cross-validation and hyperparameter tuning (e.g. `GridSearchCV`/`RandomizedSearchCV`) for a fairer, more robust comparison.
- Try additional models (e.g. Support Vector Machine, Gradient Boosting/XGBoost).
- Add SHAP-based explainability for individual predictions, beyond global feature importance.
- Test calibration of predicted probabilities (important if the model's confidence scores would be used to prioritize review).
- Validate on an independent, external dataset if one becomes available.

## How to Run the Project

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/predictive-modeling-ml.git
cd predictive-modeling-ml

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline as a script (regenerates results/ from scratch)
python src/model.py

# 5. Or explore interactively
jupyter notebook notebooks/predictive_modeling.ipynb
```

No dataset download is required — the data loads directly from scikit-learn.

## Project Structure

```text
predictive-modeling-ml/
│
├── data/
│   └── README.md                # Dataset source & description (no raw data committed)
│
├── notebooks/
│   └── predictive_modeling.ipynb  # Full, executed EDA-to-evaluation walkthrough
│
├── src/
│   └── model.py                 # Reusable, importable pipeline (load/preprocess/train/evaluate/plot)
│
├── results/
│   ├── confusion_matrix.png     # Best model's confusion matrix
│   ├── roc_curve.png            # ROC curves for all three models
│   ├── feature_importance.png   # Random Forest feature importances
│   ├── model_comparison.csv     # Metrics table (generated, not hand-typed)
│   └── metrics.json             # Same metrics in JSON form
│
├── README.md
├── requirements.txt
└── .gitignore
```
