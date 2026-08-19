"""One-call JSON planner for the small set of locally available tools."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class PlanStep:
    tool: str
    action: str
    arguments: dict[str, Any]


class PlanError(ValueError):
    pass


class Planner:
    def __init__(self, llm: Callable[[str], str], max_steps: int = 6) -> None:
        self.llm, self.max_steps = llm, max_steps

    def plan(self, question: str, feedback: str | None = None) -> list[PlanStep]:
        prompt = (
            "Return JSON only: {\"steps\":[...]}. Available steps are data/query with "
            "{expression}, data/describe with optional {columns}, data/filter with {filters}, "
            "data/groupby_aggregate with {by,aggregations}, model/predict with {customer_id} "
            "or {features}, and model/batch_predict with {records}. Plan only; do not calculate "
            "or state results. Data queries use only df plus approved pandas read methods. "
            f"Question: {question}"
        )
        if feedback:
            prompt += f" Previous plan failed verification: {feedback}. Correct it."
        try:
            payload = json.loads(self.llm(prompt))
            raw_steps = payload["steps"]
        except (TypeError, ValueError, KeyError) as exc:
            raise PlanError("LLM did not return a valid plan JSON object") from exc
        if not isinstance(raw_steps, list) or not raw_steps or len(raw_steps) > self.max_steps:
            raise PlanError(f"Plan must contain between 1 and {self.max_steps} steps")
        steps = []
        for step in raw_steps:
            if not isinstance(step, dict) or step.get("tool") not in {"data", "model"}:
                raise PlanError("Plan contains an unknown tool")
            action, arguments = step.get("action"), step.get("arguments", {})
            if not isinstance(action, str) or not isinstance(arguments, dict):
                raise PlanError("Each plan step needs action and arguments")
            steps.append(PlanStep(step["tool"], action, arguments))
        return steps
