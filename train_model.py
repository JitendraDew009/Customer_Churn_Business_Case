"""Train the churn model from a CSV file or the project database."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import joblib
import pandas as pd
from sqlalchemy import URL, create_engine, text

from churn_project import ChurnModelBundle, train_churn_model


def read_database_data() -> pd.DataFrame:
    """Read the project table over a read-only MySQL session."""
    required_variables = {
        "host": "TELECOM_DB_HOST",
        "port": "TELECOM_DB_PORT",
        "database": "TELECOM_DB_NAME",
        "username": "TELECOM_DB_USER",
        "password": "TELECOM_DB_PASSWORD",
    }
    settings = {key: os.environ.get(variable) for key, variable in required_variables.items()}
    missing = [
        variable for key, variable in required_variables.items() if not settings[key]
    ]
    if missing:
        raise ValueError(
            "Database mode requires these environment variables: "
            + ", ".join(missing)
        )

    url = URL.create(
        "mysql+pymysql",
        username=settings["username"],
        password=settings["password"],
        host=settings["host"],
        port=int(settings["port"]),
        database=settings["database"],
        query={"charset": "utf8mb4"},
    )
    engine = create_engine(url, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            connection.exec_driver_sql("SET SESSION TRANSACTION READ ONLY")
            return pd.read_sql_query(
                text("SELECT * FROM telecom_churn_data"), connection
            )
    finally:
        engine.dispose()


def train_and_save(data: pd.DataFrame, output_dir: Path) -> ChurnModelBundle:
    """Train the model and write only model and aggregate analysis artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    bundle, metrics, importance = train_churn_model(data)
    joblib.dump(bundle, output_dir / "churn_model.joblib")
    (reports_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    importance.to_csv(reports_dir / "permutation_importance.csv", index=False)

    print(f"Trained on {len(data):,} rows; held-out test rows: {metrics['test_rows']:,}.")
    print(f"Test ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"Test average precision: {metrics['average_precision']:.3f}")
    print(
        f"Test churn prevalence: {metrics['test_churn_rate']:.3f}; "
        f"average-precision lift: {metrics['average_precision_lift']:.3f}"
    )
    print(
        f"Decision threshold: {metrics['threshold']:.3f} "
        f"({metrics['threshold_method']}); test flag rate: {metrics['flag_rate']:.1%}"
    )
    if metrics["roc_auc"] < 0.55 and metrics["average_precision_lift"] < 1.1:
        print(
            "WARNING: held-out performance is near random chance; "
            "do not use scores alone to target retention campaigns."
        )
    print(f"Model and aggregate reports saved under: {output_dir.resolve()}")
    return bundle


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="Path to a labeled CSV dataset.")
    source.add_argument(
        "--database",
        action="store_true",
        help="Read telecom_churn_data using TELECOM_DB_* environment variables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory for the model and aggregate reports (default: models).",
    )
    args = parser.parse_args(argv)

    data = (
        pd.read_csv(args.data)
        if args.data is not None
        else read_database_data()
    )
    train_and_save(data, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
