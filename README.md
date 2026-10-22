# Telco Customer Churn Prediction

CM2604 Machine Learning coursework (Part 01). The project predicts whether a telecom customer will **churn** (cancel their service) using three classifiers: **K-Nearest Neighbours**, **Logistic Regression** and **Random Forest**.

## Dataset

- **Telco Customer Churn** (IBM sample dataset, available on Kaggle): `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- 7,043 customers and 21 columns (contract, services, charges, tenure, demographics)
- Target: `Churn` (Yes = 26.5%, No = 73.5%), so the classes are imbalanced

## Project structure

| Path | Contents |
|---|---|
| `churn.ipynb` | Full workflow: cleaning, EDA, preprocessing, model training, tuning, evaluation, feature engineering and fairness checks |
| `figures/` | Charts used in the report, plus `results_table.csv` |
| `requirements.txt` | Python packages and versions used |

## How to run

Requires Python 3.14. In Windows cmd:

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Open `churn.ipynb` in VS Code or Jupyter, select the `venv` kernel and click **Run All**. The grid search takes a few minutes.

## Method

1. **Cleaning:** 11 blank `TotalCharges` values (all new customers with tenure 0) set to 0, `customerID` dropped, target encoded as 1/0.
2. **EDA:** class balance, churn rate for every feature, numeric distributions, outlier check and correlation analysis.
3. **Split and preprocessing:** stratified 80/20 train/test split. `StandardScaler` and `OneHotEncoder` inside a scikit-learn `Pipeline`, so they are fitted on training data only (no data leakage).
4. **Experiment 1:** all three models with default hyperparameters.
5. **Experiment 2:** hyperparameter tuning with `GridSearchCV` (5-fold stratified cross-validation, scored on F1).
6. **Experiment 3:** engineered features (`NumServices`, `TenureGroup`) compared against the original features.
7. **Fairness checks:** model behaviour compared across gender and senior-citizen groups.

F1, recall and ROC-AUC are used instead of accuracy alone because of the class imbalance.

## Results (test set, tuned models)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| KNN | 0.783 | 0.591 | 0.588 | 0.590 | 0.832 |
| Logistic Regression | 0.741 | 0.508 | 0.786 | 0.617 | 0.841 |
| **Random Forest** | 0.754 | 0.525 | **0.791** | **0.631** | **0.844** |

The tuned Random Forest performs best. The full table, including default settings, is in `figures/results_table.csv`.

## Author

Jaruban Yogarathnam
