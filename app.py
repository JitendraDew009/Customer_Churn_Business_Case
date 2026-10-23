"""Streamlit interface for ranking telecom customers by churn risk."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from churn_project import ChurnModelBundle, score_customers

MODEL_PATH = Path("models/churn_model.joblib")
METRICS_PATH = Path("models/reports/test_metrics.json")
IMPORTANCE_PATH = Path("models/reports/permutation_importance.csv")

st.set_page_config(page_title="No-Churn Customer Risk", layout="wide")
st.title("No-Churn Telecom | Customer Churn Risk")
st.write(
    "Score customer records for retention outreach. "
    "The risk score is a model estimate, not a guarantee of churn."
)

if not MODEL_PATH.is_file():
    st.error(
        "No trained model was found. Train one with "
        "`python train_model.py --database` or `python train_model.py --data <file.csv>`."
    )
    st.stop()

bundle = joblib.load(MODEL_PATH)
if not isinstance(bundle, ChurnModelBundle):
    st.error("The saved model has an unexpected format. Retrain the model.")
    st.stop()

metrics = None
if METRICS_PATH.is_file():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    metric_columns = st.columns(5)
    metric_columns[0].metric("Held-out ROC-AUC", f"{metrics['roc_auc']:.3f}")
    metric_columns[1].metric("Held-out average precision", f"{metrics['average_precision']:.3f}")
    metric_columns[2].metric("No-skill AP reference", f"{metrics['test_churn_rate']:.3f}")
    metric_columns[3].metric("Held-out precision", f"{metrics['precision']:.1%}")
    metric_columns[4].metric("Held-out recall", f"{metrics['recall']:.1%}")

    if metrics["roc_auc"] < 0.55 and metrics["average_precision_lift"] < 1.1:
        st.warning(
            "Held-out results are near random-chance performance. "
            "Do not use these scores alone to allocate retention offers; "
            "the current dataset does not show reliable churn predictors."
        )

st.caption(
    f"Validation threshold: {bundle.threshold:.1%} "
    f"({bundle.threshold_method.replace('_', ' ')}). "
    "Risk scores are model scores and are not calibrated probabilities."
)
threshold_percent = st.slider(
    "Flag customers at or above this risk-score threshold",
    min_value=0,
    max_value=100,
    value=int(round(bundle.threshold * 100)),
    help="Adjusting this changes the number flagged, not the model's predictive performance.",
)

uploaded_file = st.file_uploader("Upload customer records (CSV)", type=["csv"])
with st.expander("Required columns and blank template"):
    st.write(
        "Upload the same predictor columns used to train the model. "
        "The target and customer identifiers are not used as predictors."
    )
    st.code(", ".join(bundle.feature_columns))
    template = pd.DataFrame(columns=bundle.feature_columns)
    st.download_button(
        "Download blank CSV template",
        template.to_csv(index=False).encode("utf-8"),
        file_name="customer_scoring_template.csv",
        mime="text/csv",
    )

if uploaded_file is not None:
    try:
        customers = pd.read_csv(uploaded_file)
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"Could not read the uploaded CSV: {error}")
        st.stop()

    try:
        scored = score_customers(customers, bundle, threshold=threshold_percent / 100)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    churn_count = int(scored["CHURN-FLAG"].sum())
    high_risk_count = int((scored["CHURN-RISK-SCORE"] >= 70).sum())
    overview = st.columns(3)
    overview[0].metric("Customers scored", f"{len(scored):,}")
    overview[1].metric("Flagged for retention", f"{churn_count:,}")
    overview[2].metric("Risk score >= 70", f"{high_risk_count:,}")

    st.subheader("Scored customer records")
    st.dataframe(scored, use_container_width=True, hide_index=True)
    st.download_button(
        "Download scored customers",
        scored.to_csv(index=False).encode("utf-8"),
        file_name="customers_with_churn_risk.csv",
        mime="text/csv",
        type="primary",
    )

low_signal = (
    metrics is not None
    and metrics["roc_auc"] < 0.55
    and metrics["average_precision_lift"] < 1.1
)
if IMPORTANCE_PATH.is_file() and not low_signal:
    importance = pd.read_csv(IMPORTANCE_PATH).head(15).sort_values("importance_mean")
    st.subheader("Leading churn drivers")
    st.caption("Permutation importance on held-out customers; importance is predictive, not causal.")
    st.bar_chart(importance.set_index("feature")["importance_mean"])
elif low_signal:
    st.info("Feature rankings are omitted because the held-out model showed no reliable predictive signal.")
