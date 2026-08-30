# Customer Churn Prediction

Predicting which telecom customers are likely to cancel their subscription, using the IBM Telco Customer Churn dataset.

## Problem Statement

Customer churn is expensive — acquiring a new customer costs far more than retaining an existing one. If a company can identify which customers are at risk of leaving *before* they leave, it can target retention offers where they'll actually matter instead of spending on customers who were never going to churn anyway.

This project builds a model that predicts churn from a customer's account details, contract type, and services subscribed.

## Objectives

- Explore the dataset to understand what drives churn
- Clean and prepare the data for modeling
- Train and compare a baseline model against a more complex one
- Evaluate the models properly, given that churn is an imbalanced outcome
- Identify which features matter most, for a non-technical audience

## Dataset

[IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) — 7,043 customers, 21 columns covering demographics, account details (tenure, contract type, payment method), subscribed services (phone, internet, streaming, tech support), billing (monthly/total charges), and whether the customer churned.

## Technologies Used

Python, pandas, NumPy, matplotlib, seaborn, scikit-learn

## Project Structure

```
customer-churn-prediction/
├── README.md
├── data/
│   └── Telco-Customer-Churn.csv
├── notebooks/
│   └── churn_prediction.ipynb
├── results/
│   └── (charts exported from the notebook)
├── requirements.txt
└── .gitignore
```

## Data Cleaning

`TotalCharges` was stored as text rather than a number. Digging into the blank values showed they all belonged to customers with `tenure == 0` — brand new customers who hadn't been billed yet — so those were filled with 0 rather than dropped.

## Exploratory Data Analysis

- **Churn rate:** about 27% of customers in the dataset churned — an imbalanced target, which matters for how the models are evaluated later.
- **Contract type** is one of the strongest signals: month-to-month customers churn far more than customers on one- or two-year contracts, which makes sense since there's no penalty for leaving.
- **Tenure**: churn is heavily concentrated in the first several months. Customers who stay past year one are much less likely to leave.
- `TotalCharges` is highly correlated with `tenure` (it's roughly `tenure × MonthlyCharges`), so it doesn't add much independent signal.

| Churn distribution | Churn by contract type | Tenure by churn |
|---|---|---|
| ![Churn distribution](results/churn_distribution.png) | ![Churn by contract](results/churn_by_contract.png) | ![Tenure by churn](results/tenure_by_churn.png) |

## Feature Engineering

- Binary Yes/No columns (`Partner`, `Dependents`, `PaperlessBilling`, etc.) were label encoded.
- Categorical columns with more than two values (`Contract`, `InternetService`, `PaymentMethod`, etc.) were one-hot encoded so the models don't assume a false order between categories.
- Numeric columns (`tenure`, `MonthlyCharges`, `TotalCharges`) were standardized before Logistic Regression — without scaling, the solver failed to converge cleanly, since these columns range into the thousands while the one-hot columns are just 0/1.

## Methodology

An 80/20 train/test split was used, stratified on the churn label to keep the same class balance in both sets.

## Models Used

- **Logistic Regression** — the baseline. Simple, fast, and interpretable, so it's a reasonable bar for a more complex model to beat.
- **Random Forest** — tried next to see whether modeling non-linear feature interactions improves on the linear baseline.

## Model Evaluation

Accuracy alone is misleading here — a model that predicted "No churn" for everyone would already score ~73% accuracy without being useful. Precision, recall, and ROC-AUC give a fuller picture of how well each model actually separates churners from non-churners.

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1 (Churn) | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.81 | 0.66 | 0.56 | 0.60 | **0.842** |
| Random Forest | 0.81 | 0.67 | 0.53 | 0.59 | 0.841 |

![ROC curve comparison](results/roc_comparison.png)

![Confusion matrix](results/confusion_matrix.png)

## Results

The two models perform almost identically. Random Forest's ability to capture non-linear interactions doesn't add meaningful lift here, likely because the strongest churn signals (contract type, tenure) are already close to linearly related to the outcome. Given that, Logistic Regression is the more sensible choice in practice — same performance, but easier to explain to a non-technical stakeholder.

## Key Findings

![Feature importance](results/feature_importance.png)

`tenure`, `TotalCharges`, `MonthlyCharges`, and `Contract` type are consistently the strongest predictors. The highest-risk group is new customers on month-to-month contracts paying high monthly charges — which matches a simple, actionable retention strategy: prioritize incentivizing longer contracts, especially in a customer's first few months.

## Limitations

- **Recall on churners is moderate (~53–56%)** — the models miss close to half of customers who actually churn. For a real retention campaign, this would need improving before relying on it.
- The dataset is a **snapshot**, not a time series — it can't show *when* a customer is about to churn, only whether they're at elevated risk.
- No hyperparameter tuning was done beyond a couple of reasonable defaults (`max_depth=10` for the Random Forest); a grid/random search would likely squeeze out a bit more performance.
- Class imbalance wasn't directly addressed (e.g. with class weighting or resampling), which likely holds recall back on the minority (churn) class.

## Future Improvements

- Address class imbalance directly (`class_weight="balanced"`, SMOTE) to try to lift recall
- Hyperparameter tuning with cross-validation
- Try gradient boosting (XGBoost/LightGBM) as a stronger comparison model
- Threshold tuning — the default 0.5 cutoff isn't necessarily optimal if false negatives (missed churners) are more costly than false positives

## How to Run the Project

```bash
git clone https://github.com/amirajsingh6/customer-churn-prediction.git
cd customer-churn-prediction

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook notebooks/churn_prediction.ipynb
```
