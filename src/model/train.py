"""
Train the churn model pipeline and export the artifact used by predict.py.

Run as a script:
    python -m src.model.train

Produces:
    data/model_pipeline.joblib   - fitted sklearn Pipeline (preprocessing + LR)
    data/cleaned_churn.csv       - cleaned reference dataset (for customer_id lookups)
    data/metrics.json            - evaluation metrics for both candidate models
"""

from __future__ import annotations

import json
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.model.cleaning import (
    NUMERIC_FEATURES,
    get_categorical_features,
    get_feature_target_split,
    load_and_clean_data,
)

RANDOM_STATE = 42
RAW_DATA_PATH = "data/raw_churn.csv"
ARTIFACT_PATH = "data/model_pipeline.joblib"
CLEANED_DATA_PATH = "data/cleaned_churn.csv"
METRICS_PATH = "data/metrics.json"


def build_preprocessor(categorical_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
                categorical_features,
            ),
        ]
    )


def evaluate(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    proba = pipe.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "classification_report_at_0.5": classification_report(
            y_test, preds, digits=3, output_dict=True
        ),
    }


def main() -> None:
    df = load_and_clean_data(RAW_DATA_PATH)
    X, y = get_feature_target_split(df)
    categorical_features = get_categorical_features(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(categorical_features)

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    metrics = {}
    fitted_pipelines = {}
    for name, clf in candidates.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        pipe.fit(X_train, y_train)
        metrics[name] = evaluate(pipe, X_test, y_test)
        fitted_pipelines[name] = pipe

    # Decision: ship logistic_regression. See notebook for full justification —
    # ROC-AUC is within ~0.0005 of random_forest and PR-AUC within ~0.018,
    # while LR gives honest per-prediction coefficient-based explanations
    # (needed for the `top_factors` output) without an extra explainability
    # library, and is cheaper to call repeatedly inside batch/aggregate queries.
    shipped_model_name = "logistic_regression"
    shipped_pipeline = fitted_pipelines[shipped_model_name]

    os.makedirs("data", exist_ok=True)
    joblib.dump(
        {
            "pipeline": shipped_pipeline,
            "model_name": shipped_model_name,
            "feature_columns": list(X.columns),
            "categorical_features": categorical_features,
            "numeric_features": NUMERIC_FEATURES,
        },
        ARTIFACT_PATH,
    )
    df.to_csv(CLEANED_DATA_PATH)

    metrics["shipped_model"] = shipped_model_name
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved pipeline artifact to {ARTIFACT_PATH}")
    print(f"Saved cleaned reference data to {CLEANED_DATA_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    print()
    print("ROC-AUC  — logreg:", metrics["logistic_regression"]["roc_auc"],
          " rf:", metrics["random_forest"]["roc_auc"])
    print("PR-AUC   — logreg:", metrics["logistic_regression"]["pr_auc"],
          " rf:", metrics["random_forest"]["pr_auc"])


if __name__ == "__main__":
    main()
