import numpy as np
import pandas as pd
import pytest

from churn_project import (
    ChurnModelBundle,
    normalize_target,
    prepare_features,
    score_customers,
    train_churn_model,
)


def test_normalize_target_accepts_common_yes_no_labels():
    labels = pd.Series([" YES ", "no", "True", "FALSE", 1, 0])

    result = normalize_target(labels)

    assert result.tolist() == [1, 0, 1, 0, 1, 0]
    assert str(result.dtype) == "int8"


def test_normalize_target_rejects_missing_or_unknown_labels():
    with pytest.raises(ValueError, match="unsupported labels"):
        normalize_target(pd.Series(["yes", "maybe", None]))


def test_prepare_features_excludes_target_and_phone():
    data = pd.DataFrame(
        {
            "Phone": ["111", "222"],
            "State": ["CA", "NY"],
            "Area Code": [415, 212],
            "Churn": ["No", "Yes"],
        }
    )

    features, target = prepare_features(data)

    assert features.columns.tolist() == ["State", "Area Code"]
    assert target.tolist() == [0, 1]


def test_prepare_features_excludes_ids_and_normalizes_registration_dates():
    data = pd.DataFrame(
        {
            "customer_id": [101, 102],
            "pincode": [400001, 400002],
            "date_of_registration": ["2020-01-01", "2020-01-02"],
            "Churn": [0, 1],
        }
    )

    features, _ = prepare_features(data)

    assert features.columns.tolist() == ["date_of_registration"]
    assert features["date_of_registration"].diff().iloc[1] == 1


def test_training_and_scoring_add_binary_flags_and_risk_scores():
    rng = np.random.default_rng(17)
    rows = 160
    international_plan = rng.choice(["Yes", "No"], size=rows, p=[0.25, 0.75])
    day_minutes = rng.normal(180, 35, size=rows).clip(30, 350)
    churn_probability = np.where(
        international_plan == "Yes",
        0.72,
        np.where(day_minutes > 210, 0.42, 0.08),
    )
    churn = np.where(rng.random(rows) < churn_probability, "Yes", "No")
    data = pd.DataFrame(
        {
            "Phone": [f"p{i}" for i in range(rows)],
            "State": rng.choice(["CA", "NY", "TX"], size=rows),
            "Area Code": rng.choice([415, 212, 512], size=rows),
            "International Plan": international_plan,
            "Day Mins": day_minutes,
            "Churn": churn,
        }
    )

    bundle, metrics, importance = train_churn_model(data)
    scored = score_customers(data.drop(columns="Churn").head(12), bundle)

    assert len(scored) == 12
    assert set(scored["CHURN-FLAG"].unique()).issubset({0, 1})
    assert scored["CHURN-RISK-SCORE"].between(0, 100).all()
    assert 0 <= bundle.threshold <= 1
    assert 0 <= metrics["roc_auc"] <= 1
    assert metrics["test_rows"] > 0
    assert not importance.empty


def test_score_customers_reports_missing_predictors():
    bundle = ChurnModelBundle(model=None, feature_columns=["Day Mins"], threshold=0.5)

    with pytest.raises(ValueError, match="Missing required predictor columns"):
        score_customers(pd.DataFrame({"State": ["CA"]}), bundle)


def test_score_customers_accepts_a_campaign_threshold_override():
    class FixedProbabilityModel:
        def predict_proba(self, features):
            probabilities = np.array([0.4, 0.8])
            return np.column_stack((1 - probabilities, probabilities))

    bundle = ChurnModelBundle(
        model=FixedProbabilityModel(),
        feature_columns=["age"],
        threshold=0.5,
    )
    customers = pd.DataFrame({"age": [20, 30]})

    scored = score_customers(customers, bundle, threshold=0.9)

    assert scored["CHURN-FLAG"].tolist() == [0, 0]


def test_score_customers_rejects_out_of_range_threshold():
    bundle = ChurnModelBundle(model=None, feature_columns=["age"], threshold=0.5)

    with pytest.raises(ValueError, match="between 0 and 1"):
        score_customers(pd.DataFrame({"age": [20]}), bundle, threshold=1.1)
