"""Execute a validated plan and write all tool output to the fact ledger."""

from __future__ import annotations

from typing import Any

from src.agent.fact_ledger import FactLedger
from src.agent.planner import PlanStep
from src.tools.data_tools import DataTools
from src.tools.model_tool import batch_predict_churn_risk, predict_risk


class Executor:
    def __init__(self, data_tools: DataTools) -> None:
        self.data_tools = data_tools

    def execute(self, steps: list[PlanStep], ledger: FactLedger) -> None:
        for step in steps:
            try:
                value = self._run(step)
                ledger.add(step.tool, value, f"{step.tool}/{step.action}")
            except Exception as exc:  # Error is evidence for the verifier, not an answer.
                ledger.add(step.tool, {"error": str(exc)}, f"{step.tool}/{step.action}")

    def _run(self, step: PlanStep) -> Any:
        if step.tool == "data":
            actions = {"query": self.data_tools.query, "describe": self.data_tools.describe,
                       "filter": self.data_tools.filter, "groupby_aggregate": self.data_tools.groupby_aggregate}
            if step.action not in actions:
                raise ValueError(f"Unknown data action: {step.action}")
            return actions[step.action](**step.arguments)
        if step.tool == "model":
            if step.action == "predict":
                return predict_risk(**step.arguments)
            if step.action == "batch_predict":
                return batch_predict_churn_risk(**step.arguments)
            raise ValueError(f"Unknown model action: {step.action}")
        raise ValueError(f"Unknown tool: {step.tool}")
