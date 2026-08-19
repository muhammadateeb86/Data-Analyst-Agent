"""
Standalone churn prediction callable, independent of the training notebook.

Loads the fitted pipeline artifact produced by `src.model.train` and exposes:

    predict_churn_risk(customer_id=None, features=None) -> {
        "risk_score": float,          # P(churn) in [0, 1]
        "top_factors": [ {feature, direction, contribution}, ... ],
        "source": "customer_id" | "hypothetical",
        "customer_id": str | None,
    }

Exactly one of `customer_id` or `features` must be given:
- `customer_id`: looks up the customer's real feature values from the
  cleaned reference dataset (data/cleaned_churn.csv).
- `features`: a dict of raw feature values (same schema as the training
  columns) for a hypothetical/what-if data point, e.g. to project a
  customer's risk under a different contract type.

`top_factors` is a genuine per-prediction explanation, not a fixed global
importance list: for a linear model, logit = intercept + sum(coef_i *
encoded_value_i), so each encoded feature's signed contribution to *this*
specific prediction can be computed exactly and ranked by |contribution|.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib

ARTIFACT_PATH = "data/model_pipeline.joblib"
CLEANED_DATA_PATH = "data/cleaned_churn.csv"

_artifact = None
_reference_df = None


class PredictionInputError(ValueError):
    """Raised when the caller's input can't be turned into a valid prediction."""


def _load_artifact():
    global _artifact
    if _artifact is None:
        _artifact = joblib.load(ARTIFACT_PATH)
    return _artifact


def _load_reference_df():
    global _reference_df
    if _reference_df is None:
        _reference_df = pd.read_csv(CLEANED_DATA_PATH).set_index("customerID")
    return _reference_df


def _row_for_customer(customer_id: str) -> pd.DataFrame:
    df = _load_reference_df()
    if customer_id not in df.index:
        raise PredictionInputError(f"Unknown customer_id: {customer_id!r}")
    row = df.loc[[customer_id]].drop(columns=["Churn"])
    return row


def _row_for_hypothetical(features: dict, feature_columns: list[str]) -> pd.DataFrame:
    missing = [c for c in feature_columns if c not in features]
    if missing:
        raise PredictionInputError(
            f"Missing required feature(s) for a hypothetical prediction: {missing}"
        )
    extra = [c for c in features if c not in feature_columns]
    if extra:
        raise PredictionInputError(
            f"Unknown feature(s) not in the model schema: {extra}"
        )
    row = pd.DataFrame([features], columns=feature_columns)
    return row


def _compute_top_factors(pipeline, row: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """Per-prediction signed contributions for a linear (logistic) model."""
    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]

    if not hasattr(clf, "coef_"):
        # Guard: this explanation method only makes sense for a linear model.
        # If the shipped model ever changes to a non-linear one, fail loudly
        # instead of returning a silently wrong "explanation".
        raise NotImplementedError(
            "Per-prediction linear contributions are only implemented for "
            "linear classifiers (found: %s)" % type(clf).__name__
        )

    encoded = prep.transform(row)
    encoded = np.asarray(encoded.todense()) if hasattr(encoded, "todense") else np.asarray(encoded)
    feature_names = prep.get_feature_names_out()
    coefs = clf.coef_[0]

    contributions = encoded[0] * coefs
    order = np.argsort(-np.abs(contributions))[:top_n]

    factors = []
    for i in order:
        contribution = float(contributions[i])
        if abs(contribution) < 1e-9:
            continue
        factors.append(
            {
                "feature": feature_names[i],
                "direction": "increases_risk" if contribution > 0 else "decreases_risk",
                "contribution": contribution,
            }
        )
    return factors


def predict_churn_risk(customer_id: str | None = None, features: dict | None = None) -> dict:
    if (customer_id is None) == (features is None):
        raise PredictionInputError(
            "Provide exactly one of customer_id or features, not both/neither."
        )

    artifact = _load_artifact()
    pipeline = artifact["pipeline"]
    feature_columns = artifact["feature_columns"]

    if customer_id is not None:
        row = _row_for_customer(customer_id)
        source = "customer_id"
    else:
        row = _row_for_hypothetical(features, feature_columns)
        source = "hypothetical"

    risk_score = float(pipeline.predict_proba(row)[:, 1][0])
    top_factors = _compute_top_factors(pipeline, row)

    return {
        "risk_score": risk_score,
        "top_factors": top_factors,
        "source": source,
        "customer_id": customer_id,
    }
