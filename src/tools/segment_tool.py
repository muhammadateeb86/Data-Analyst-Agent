"""Aggregate predicted churn risk across a segment of the dataset.

This closes a real gap: "which Contract type has the highest average
predicted churn risk?" needs the model to be applied to many rows at once,
then grouped. A single-shot LLM plan can't first fetch matching customer IDs
and then feed them into a batch-predict step (there's no intermediate
re-planning in this agent), so without a dedicated tool the planner has
nothing valid to call and fails with "Plan contains an unknown tool" (see
the transcript this was built against). Filtering, scoring, and grouping are
all done here, deterministically, in one tool call.
"""

from __future__ import annotations

from typing import Any

from src.model.predict import predict_risk_scores
from src.tools.data_tools import DataTools, _json_value


def segment_churn_risk(data_tools: DataTools, by: str | list[str], filters: dict[str, Any] | None = None) -> dict[str, Any]:
    groups = [by] if isinstance(by, str) else list(by)
    data_tools._require_columns(groups)
    frame = data_tools.filtered_frame(filters)
    if frame.empty:
        return {"type": "dataframe", "rows": [], "row_count": 0, "truncated": False}

    scores = predict_risk_scores(frame)
    working = frame[groups].copy()
    working["predicted_churn_risk"] = scores
    summary = (
        working.groupby(groups, dropna=False)["predicted_churn_risk"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_predicted_risk", "count": "customer_count"})
        .sort_values("avg_predicted_risk", ascending=False)
    )
    return _json_value(summary)
