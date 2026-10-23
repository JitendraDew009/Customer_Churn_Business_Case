"""Training and scoring utilities for the telecom churn project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET_COLUMN = "Churn"
PREDICTION_COLUMN = "CHURN-FLAG"
RISK_SCORE_COLUMN = "CHURN-RISK-SCORE"
RANDOM_STATE = 42


def _is_identifier(column: Any) -> bool:
    normalized = "".join(
        character.lower() if character.isalnum() else "_"
        for character in str(column).strip()
    ).strip("_")
    return normalized in {"phone", "pincode", "postal_code", "zip_code", "id"} or normalized.endswith(
        "_id"
    )


def _normalize_feature_values(features: pd.DataFrame) -> pd.DataFrame:
    normalized = features.copy()
    for column in normalized.columns:
        if "date" not in str(column).casefold():
            continue
        parsed = pd.to_datetime(normalized[column], errors="coerce", utc=True)
        invalid = normalized[column].notna() & parsed.isna()
        if invalid.any():
            raise ValueError(f"Invalid date values found in {column!r}.")
        normalized[column] = (
            parsed - pd.Timestamp("1970-01-01", tz="UTC")
        ).dt.total_seconds() / 86_400
    return normalized


@dataclass
class ChurnModelBundle:
    model: Pipeline
    feature_columns: list[str]
    threshold: float
    threshold_method: str = "validation_f2"


def normalize_target(target: pd.Series) -> pd.Series:
    """Convert common yes/no and boolean labels into binary churn values."""
    normalized = target.astype("string").str.strip().str.casefold()
    label_map = {
        "yes": 1,
        "true": 1,
        "1": 1,
        "1.0": 1,
        "no": 0,
        "false": 0,
        "0": 0,
        "0.0": 0,
    }
    result = normalized.map(label_map)
    if result.isna().any():
        invalid = sorted(normalized[result.isna()].dropna().unique().tolist())
        raise ValueError(
            f"{TARGET_COLUMN} contains missing or unsupported labels: {invalid[:10]}"
        )
    return result.astype("int8")


def prepare_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Validate the source data and separate target from model-ready features."""
    if data.empty:
        raise ValueError("The dataset is empty.")
    if not data.columns.is_unique:
        raise ValueError("The dataset contains duplicate column names.")

    target_columns = [
        column
        for column in data.columns
        if str(column).strip().casefold() == TARGET_COLUMN.casefold()
    ]
    if len(target_columns) != 1:
        raise ValueError(f"The dataset must contain exactly one {TARGET_COLUMN!r} column.")

    target_column = target_columns[0]
    y = normalize_target(data[target_column])
    excluded = {
        target_column,
        *[column for column in data.columns if _is_identifier(column)],
    }
    X = _normalize_feature_values(data.drop(columns=list(excluded)))
    if X.shape[1] == 0:
        raise ValueError("The dataset has no usable predictor columns.")
    if X.columns.has_duplicates:
        raise ValueError("Predictor column names must be unique.")
    return X, y


def build_model(features: pd.DataFrame) -> Pipeline:
    """Build a preprocessing and class-balanced random forest pipeline."""
    categorical_columns = [
        column
        for column in features.columns
        if (
            pd.api.types.is_object_dtype(features[column])
            or pd.api.types.is_string_dtype(features[column])
            or isinstance(features[column].dtype, pd.CategoricalDtype)
            or str(column).strip().casefold() == "area code"
        )
    ]
    numeric_columns = [
        column for column in features.columns if column not in categorical_columns
    ]

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="median", keep_empty_features=True),
                        )
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent", keep_empty_features=True
                            ),
                        ),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                categorical_columns,
            )
        )

    preprocessing = ColumnTransformer(transformers=transformers)
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=2,
    )
    return Pipeline(
        steps=[("preprocessing", preprocessing), ("classifier", classifier)]
    )


def select_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Select a recall-oriented decision threshold using validation F2."""
    if y_true.nunique() != 2:
        raise ValueError("Threshold selection requires both churn classes.")
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    scores = [
        fbeta_score(y_true, probabilities >= threshold, beta=2, zero_division=0)
        for threshold in candidates
    ]
    best_index = max(
        range(len(candidates)), key=lambda index: (scores[index], candidates[index])
    )
    return float(candidates[best_index])


def evaluate_predictions(
    y_true: pd.Series, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Calculate threshold-dependent and threshold-independent test metrics."""
    if y_true.nunique() != 2:
        raise ValueError("Evaluation requires both churn classes.")
    if len(y_true) != len(probabilities):
        raise ValueError("Labels and predicted probabilities must have equal lengths.")

    predictions = probabilities >= threshold
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "f2": float(fbeta_score(y_true, predictions, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": matrix.tolist(),
        "test_rows": int(len(y_true)),
        "test_churn_rate": float(y_true.mean()),
    }


def train_churn_model(data: pd.DataFrame) -> tuple[ChurnModelBundle, dict[str, Any], pd.DataFrame]:
    """Split, train, tune the decision threshold, and report held-out results."""
    features, target = prepare_features(data)
    if target.value_counts().min() < 3:
        raise ValueError("At least three examples of each churn class are required.")

    X_development, X_test, y_development, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        stratify=target,
        random_state=RANDOM_STATE,
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=0.20,
        stratify=y_development,
        random_state=RANDOM_STATE,
    )

    evaluation_model = build_model(X_train)
    evaluation_model.fit(X_train, y_train)
    validation_probabilities = evaluation_model.predict_proba(X_validation)[:, 1]
    validation_auc = float(roc_auc_score(y_validation, validation_probabilities))
    if validation_auc < 0.55:
        threshold = 0.5
        threshold_method = "default_0.5_due_to_near_chance_validation_auc"
    else:
        threshold = select_threshold(y_validation, validation_probabilities)
        threshold_method = "validation_f2"

    test_probabilities = evaluation_model.predict_proba(X_test)[:, 1]
    metrics = evaluate_predictions(y_test, test_probabilities, threshold)
    metrics["validation_roc_auc"] = validation_auc
    metrics["threshold_method"] = threshold_method
    metrics["flag_rate"] = float(np.mean(test_probabilities >= threshold))
    metrics["majority_class_accuracy"] = float(max(y_test.mean(), 1 - y_test.mean()))
    metrics["average_precision_lift"] = float(
        metrics["average_precision"] / metrics["test_churn_rate"]
    )

    importance_rows = X_test.sample(
        n=min(2_000, len(X_test)), random_state=RANDOM_STATE
    )
    importance_targets = y_test.loc[importance_rows.index]
    importance = permutation_importance(
        evaluation_model,
        importance_rows,
        importance_targets,
        scoring="roc_auc",
        n_repeats=3,
        random_state=RANDOM_STATE,
        n_jobs=2,
    )
    importance_frame = (
        pd.DataFrame(
            {
                "feature": features.columns,
                "importance_mean": importance.importances_mean,
                "importance_std": importance.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    model = build_model(features)
    model.fit(features, target)
    bundle = ChurnModelBundle(
        model=model,
        feature_columns=list(features.columns),
        threshold=threshold,
        threshold_method=threshold_method,
    )
    return bundle, metrics, importance_frame


def score_customers(
    data: pd.DataFrame,
    bundle: ChurnModelBundle,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Append binary churn flags and 0-100 risk scores to customer rows."""
    if data.empty:
        raise ValueError("No customer rows were provided for scoring.")
    decision_threshold = bundle.threshold if threshold is None else threshold
    if not 0 <= decision_threshold <= 1:
        raise ValueError("The decision threshold must be between 0 and 1.")

    excluded = {
        column
        for column in data.columns
        if str(column).strip().casefold() == TARGET_COLUMN.casefold()
        or _is_identifier(column)
    }
    available = _normalize_feature_values(data.drop(columns=list(excluded)))
    missing = [column for column in bundle.feature_columns if column not in available]
    if missing:
        raise ValueError(f"Missing required predictor columns: {missing}")

    probabilities = bundle.model.predict_proba(
        available.loc[:, bundle.feature_columns]
    )[:, 1]
    result = data.copy()
    result[PREDICTION_COLUMN] = (probabilities >= decision_threshold).astype("int8")
    result[RISK_SCORE_COLUMN] = np.round(probabilities * 100, 2)
    return result
