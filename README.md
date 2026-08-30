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

The two baseline models perform almost identically. Random Forest's ability to capture non-linear interactions doesn't add meaningful lift here, likely because the strongest churn signals (contract type, tenure) are already close to linearly related to the outcome. Given that, Logistic Regression is the more sensible baseline choice — same performance, but easier to explain to a non-technical stakeholder.

## Improving Recall

The baseline only catches 56% of actual churners — a real weakness for a retention use case, where missing an at-risk customer is usually costlier than a wasted retention offer. Two fixes were tried:

| Approach | Precision (Churn) | Recall (Churn) | ROC-AUC |
|---|---|---|---|
| Baseline (threshold 0.5) | 0.66 | 0.56 | 0.842 |
| `class_weight="balanced"` | 0.50 | **0.78** | 0.842 |
| Threshold tuned to 0.3 | 0.52 | 0.76 | 0.842 |

![Precision/recall trade-off](results/threshold_tuning.png)

Class weighting turned out to be the more effective of the two — it lifts recall from 0.56 to 0.78 (catching roughly 3 in 4 churners instead of just over half), at the cost of precision dropping to 0.50. ROC-AUC barely moves in either case, which makes sense: it measures how well the model *ranks* churners above non-churners, and neither technique changes that ranking — they just shift where the decision cutoff falls. Threshold tuning alone reached a similar but slightly weaker result (0.76 recall), so class weighting is the recommended approach here, with threshold tuning available as an extra lever if the exact cost of a missed churner vs. a wasted offer is known.

![Confusion matrix, baseline model with threshold lowered to 0.3](results/confusion_matrix_tuned.png)

## Key Findings

![Feature importance](results/feature_importance.png)

`tenure`, `TotalCharges`, `MonthlyCharges`, and `Contract` type are consistently the strongest predictors. The highest-risk group is new customers on month-to-month contracts paying high monthly charges — which matches a simple, actionable retention strategy: prioritize incentivizing longer contracts, especially in a customer's first few months.

## Limitations

- **Precision drops to 0.50 in the recall-improved model** — about half of the customers flagged as "at risk" won't actually churn, so retention offers under this model would be spent inefficiently in exchange for catching more real churners.
- The dataset is a **snapshot**, not a time series — it can't show *when* a customer is about to churn, only whether they're at elevated risk.
- No hyperparameter tuning was done beyond a couple of reasonable defaults (`max_depth=10` for the Random Forest); a grid/random search would likely squeeze out a bit more performance.
- SMOTE (synthetic oversampling) wasn't tried — only class weighting and threshold tuning, which was enough to make a clear improvement without adding that extra complexity.

## Future Improvements

- Try SMOTE as a third way of addressing class imbalance, compared against class weighting
- Hyperparameter tuning with cross-validation (`GridSearchCV`)
- Try gradient boosting (XGBoost/LightGBM) as a stronger comparison model
- Pick the exact threshold using a real cost estimate (e.g. cost of a lost customer vs. cost of a retention offer) rather than the round-number 0.3 used here

## How to Run the Project

```bash
git clone https://github.com/amirajsingh6/customer-churn-prediction.git
cd customer-churn-prediction

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook notebooks/churn_prediction.ipynb
```
