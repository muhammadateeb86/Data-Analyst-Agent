"""Agent-facing wrappers around the Stage 1 churn prediction callable."""

from __future__ import annotations

from typing import Any, Iterable

from src.model.predict import predict_churn_risk


def predict_risk(customer_id: str | None = None, features: dict[str, Any] | None = None) -> dict[str, Any]:
    return predict_churn_risk(customer_id=customer_id, features=features)


def batch_predict_churn_risk(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Predict a batch of customer IDs and/or hypothetical feature records.

    Each record must contain exactly one of ``customer_id`` or ``features``.
    Errors are returned per record, so one malformed item never discards an
    otherwise useful aggregate analysis.
    """
    results: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            results.append({"index": index, "error": "Each batch item must be a dictionary"})
            continue
        try:
            prediction = predict_churn_risk(customer_id=record.get("customer_id"), features=record.get("features"))
            results.append({"index": index, **prediction})
        except (ValueError, TypeError) as exc:
            results.append({"index": index, "error": str(exc)})
    return results


# Friendly alias for callers that prefer a verb-first name.
predict_churn_risk_batch = batch_predict_churn_risk
