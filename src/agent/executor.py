"""Execute a validated plan and write all tool output to the fact ledger."""

from __future__ import annotations

from typing import Any

from src.agent.fact_ledger import FactLedger
from src.agent.planner import PlanStep
from src.tools.data_tools import DataTools
from src.tools.model_tool import batch_predict_churn_risk, predict_risk
from src.tools.segment_tool import segment_churn_risk


class Executor:
    def __init__(self, data_tools: DataTools) -> None:
        self.data_tools = data_tools

    def execute(self, steps: list[PlanStep], ledger: FactLedger) -> None:
        for step in steps:
            try:
                value = self._run(step, ledger)
                ledger.add(step.tool, value, f"{step.tool}/{step.action}")
            except Exception as exc:  # Error is evidence for the verifier, not an answer.
                ledger.add(step.tool, {"error": str(exc)}, f"{step.tool}/{step.action}")

    def _run(self, step: PlanStep, ledger: FactLedger) -> Any:
        if step.tool == "data":
            actions = {
                "count_rows": self.data_tools.count_rows,
                "describe": self.data_tools.describe,
                "filter": self.data_tools.filter,
                "groupby_aggregate": self.data_tools.groupby_aggregate,
                "query": self.data_tools.query,
                "customer_features": self.data_tools.customer_features,
            }
            if step.action not in actions:
                raise ValueError(f"Unknown data action: {step.action}")
            return actions[step.action](**step.arguments)
        if step.tool == "model":
            if step.action in {"predict_churn", "predict"}:
                return predict_risk(**step.arguments)
            if step.action == "batch_predict":
                return batch_predict_churn_risk(**step.arguments)
            if step.action == "project":
                return self._project(ledger, **step.arguments)
            if step.action == "segment_risk":
                return segment_churn_risk(self.data_tools, **step.arguments)
            raise ValueError(f"Unknown model action: {step.action}")
        raise ValueError(f"Unknown tool: {step.tool}")

    @staticmethod
    def _project(ledger: FactLedger, features_from_fact: str, overrides: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(overrides, dict):
            raise ValueError("Projection overrides must be a dictionary")
        feature_fact = ledger.get(features_from_fact).value
        if not isinstance(feature_fact, dict) or not isinstance(feature_fact.get("features"), dict):
            raise ValueError("Projection must reference a customer_features fact")
        features = dict(feature_fact["features"])
        unknown = set(overrides) - set(features)
        if unknown:
            raise ValueError(f"Projection has unknown feature override(s): {sorted(unknown)}")
        features.update(overrides)
        result = predict_risk(features=features)
        return {"base_feature_fact": features_from_fact, "overrides": dict(overrides), **result}
