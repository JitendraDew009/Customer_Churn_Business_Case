# No-Churn Telecom: customer churn prediction

This project implements the churn use cases in the PR-0017 brief:

1. Identify predictive customer characteristics with held-out permutation importance.
2. Produce a churn probability score for retention-campaign prioritization.
3. Add `CHURN-FLAG` (`1` = flagged for retention, `0` = not flagged) to customer rows.

The read-only database snapshot inspected during completion contained 243,553
rows and no missing values, rather than the 4,617 rows in the PDF metadata. Its
current columns are `customer_id`, `telecom_partner`, `gender`, `age`, `state`,
`city`, `pincode`, `date_of_registration`, `num_dependents`, `estimated_salary`,
`calls_made`, `sms_sent`, `data_used`, and `churn`. The implementation follows
the live table schema; it excludes customer identifiers and the near-unique
`pincode`, and converts registration dates to numeric days for modeling. The
target parser supports the database's numeric 0/1 labels and the PDF's yes/no
labels.

The model excludes `Phone` as an identifier, preprocesses categorical and numeric
fields in a scikit-learn pipeline, and handles class imbalance with a weighted
random forest. If validation ROC-AUC is at least 0.55, the decision threshold is
selected by maximizing F2, which gives recall more weight when prioritizing
possible churners; otherwise it falls back to 0.5 rather than flagging nearly
every customer. Reported metrics are calculated on a separate, held-out test
split. Feature importance describes predictive association, not causation.
Permutation importance is estimated on up to 2,000 held-out rows with three
repeats to keep reporting practical for the full-size database table.

The snapshot produced held-out ROC-AUC 0.502 and average precision 0.203 against
a 0.200 churn prevalence. Those near-baseline results mean the observed schema
does not currently support reliable churn targeting. The app reports this
limitation and lets an operator adjust the flag threshold, but a different
threshold cannot improve the model's ability to rank customers. Risk scores are
not calibrated probabilities.

## Setup

Use Python 3.10 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train from a CSV

The CSV must contain a `Churn` column with yes/no, true/false, or 1/0 labels.
For example:

```powershell
python train_model.py --data .\telecom_churn_data.csv
```

## Train from the project database

Configure the connection details in the current PowerShell session using
environment variables. Obtain them from your authorized project materials;
do not put passwords in source files or commit them:

```powershell
$env:TELECOM_DB_HOST = "<database host>"
$env:TELECOM_DB_PORT = "3306"
$env:TELECOM_DB_NAME = "<database name>"
$env:TELECOM_DB_USER = "<database username>"
$env:TELECOM_DB_PASSWORD = "<database password>"
python train_model.py --database
```

Database mode connects to MySQL and reads `telecom_churn_data` in a read-only
session. It does not save the downloaded customer table. Only the trained model
and aggregate evaluation/importance reports are written locally.

Training creates:

- `models/churn_model.joblib` — preprocessing and classifier pipeline, feature
  names, and selected decision threshold.
- `models/reports/test_metrics.json` — held-out metrics and confusion matrix.
- `models/reports/permutation_importance.csv` — held-out predictor importance.

## Run the prediction app

After training:

```powershell
streamlit run app.py
```

Upload a CSV containing the model's predictor columns. `Churn` may also be
present but is ignored during prediction. Customer identifiers are not used as
predictors, and the original uploaded columns are retained in the output. The
downloadable results include:

- `CHURN-FLAG`: numeric 1/0 decision at the validation-selected threshold.
- `CHURN-RISK-SCORE`: random-forest score on a 0–100 scale (not calibrated).

## Validation

Run the focused project tests with:

```powershell
python -m pytest
```

The model is a decision-support tool. Validate the threshold and campaign
capacity with the business team before using scores to allocate offers or
customer-support resources.


# Customer Churn Business Case

This project focuses on predicting customer churn in a telecom business context using machine learning. The objective is to identify customers who are likely to leave, prioritize retention efforts, and improve business decision-making through a data-driven churn risk model.

## Problem
Customer churn is one of the biggest challenges in the telecom industry. When customers leave, the company loses revenue, faces higher customer acquisition costs, and must spend additional resources on replacing lost customers. In a competitive market, identifying at-risk customers early is critical to reducing churn and improving customer retention.

Telecom companies often have large volumes of customer data, but not all variables are equally useful for predicting churn. The challenge is to analyze customer behavior and demographic data to detect patterns that signal a likely future churn event.

## Problem Faced
The main business problem was to determine which customers are at risk of churning and which ones should be prioritized for retention campaigns.

Key issues included:
- High volume of customer records with many different attributes
- Class imbalance, where churned customers are fewer than non-churned customers
- Presence of identifiers and weak predictors that do not contribute meaningfully to modeling
- Need to convert raw customer data into a model-ready dataset
- Requirement to balance predictive power with business usability
- Difficulty in making the output understandable for non-technical stakeholders

Additionally, the dataset used in this project showed weak predictive signals, which made churn detection difficult. This means that even though a model can be trained, it may not provide reliable customer-level predictions without stronger data.

## Goal
The objective of this project is to:
- Build a churn prediction model using customer-related features
- Identify underlying patterns associated with customer churn
- Estimate churn risk scores for customers
- Flag high-risk customers for retention outreach
- Provide a simple interface for business users to upload customer data and receive churn predictions

## Solution
A machine learning pipeline was developed to solve the churn prediction problem.

The solution includes:
- Data cleaning and validation
- Target label normalization for yes/no, true/false, and 1/0 churn values
- Removal of non-predictive identifiers such as customer IDs and postal codes
- Conversion of date-type fields into numeric values for modeling
- Training of a preprocessing + classification pipeline using:
  - Imputation for missing values
  - OneHotEncoder for categorical variables
  - RandomForestClassifier for churn prediction
- Use of class weighting to address imbalance
- Decision threshold tuning using F2 scoring to prioritize recall
- Evaluation using a held-out test set
- Feature importance analysis to understand major churn drivers
- Streamlit-based user interface for scoring customer records

The project also provides a trained model and value-added outputs such as:
- Binary churn flags
- Churn risk scores on a 0–100 scale
- Summary metrics for model quality
- Feature importance ranking

## Final Report
The model was trained and evaluated using a supervised learning workflow. The prepared project reports show that the model produced:

- Held-out ROC-AUC: approximately 0.502
- Held-out Average Precision: approximately 0.203
- Test churn prevalence: approximately 0.200

These values indicate that the model's performance is close to random chance and does not provide strong predictive power for reliable churn targeting with the current dataset. This is a critical result: the current data does not show a strong and dependable relationship between the available variables and churn behavior.

The model therefore:
- can generate risk scores,
- can flag a subset of customers,
- and can help with prioritization in a limited sense,

but it should not be treated as a production-grade churn forecasting system without additional data or stronger customer-level signals.

The app included in the repository allows users to:
- upload customer CSV files,
- score them using the trained model,
- view flagged customers,
- and download the output with churn predictions and risk scores.

## Conclusion
This project demonstrates a complete machine learning workflow for analyzing customer churn, from data preparation to model training, evaluation, and deployment through a simple web application.

However, the final analysis shows that the current dataset does not provide strong predictive power for churn prediction. The conclusion is that the project is useful as a business case and technical demonstration, but it is not yet strong enough for direct operational decision-making in real-world retention campaigns without better data or deeper behavioral signals.

In summary:
- The project successfully builds a churn prediction pipeline
- The workflow is realistic and scalable
- The current model is a decision-support tool rather than a highly reliable production model
- More meaningful churn predictors and richer customer data are needed for better business outcomes

This project is therefore a strong starting point for a churn analytics initiative, but it should be treated as a baseline and further improved before deployment in a high-impact customer retention strategy.
