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
    def __init__(self, llm: Callable[[str], str], max_steps: int = 6, schema: str | None = None) -> None:
        self.llm, self.max_steps, self.schema = llm, max_steps, schema

    def plan(self, question: str, feedback: str | None = None) -> list[PlanStep]:
        prompt = (
            "Return JSON only: {\"steps\":[...]}. Available steps are: "
            "data/count_rows with {} (total customer count), "
            "data/query with {expression} (df plus approved pandas read methods only), "
            "data/describe with optional {columns}, "
            "data/filter with {filters} (exact value, or a comparison dict like {\"tenure\": {\"gte\": 12}}, "
            "ops: eq/ne/gt/gte/lt/lte/in), "
            "data/groupby_aggregate with {by, aggregations} e.g. {\"by\": \"Contract\", "
            "\"aggregations\": {\"MonthlyCharges\": \"mean\"}}, "
            "model/predict with {customer_id} or {features} (current risk for one real customer, "
            "or one hypothetical/what-if record), "
            "model/batch_predict with {records} (a list of {customer_id} or {features} items you "
            "already know explicitly — never use this to enumerate a whole segment), "
            "model/segment_risk with {by, filters (optional)} — the ONLY correct way to answer "
            "'average/highest predicted risk by <column>' or any aggregate-risk-across-many-customers "
            "question; it scores every matching row and groups by `by` itself, so never pair it with "
            "a data/filter or data/query step to fetch customer IDs first. "
            "For current-versus-projected risk for ONE named customer — including a plain 'what would "
            "their risk be if X changed' question, not only explicit 'compare current vs projected' "
            "phrasing — first use data/customer_features with {customer_id}, then model/predict with "
            "{customer_id}, then model/project with {features_from_fact: 'fact_1', overrides: {...}}. "
            "Never call model/predict directly with a partial {features} dict for a named customer's "
            "hypothetical — it requires every model feature and will fail; model/project is the only "
            "way to change just one or two features for a real customer. model/project changes only "
            "keys supplied in overrides. Plan only; do not calculate or state results. For a question "
            "asking the current risk for a customer ID, use exactly model/predict with "
            "{customer_id: '<ID>'}; do not query dataframe rows first. For a 'what percentage/rate' "
            "question, use one data/query step with a boolean .mean() expression **multiplied by 100** "
            "so the fact is already in percent, e.g. "
            "(df[df['InternetService']=='Fiber optic']['Churn']=='Yes').mean()*100 — never data/filter, "
            "which only returns matching rows for the user to read, not a computed rate. For a 'which "
            "<category> has the highest/lowest rate of <condition>' question, use one data/query step "
            "that groups the boolean condition by the category and sorts descending, e.g. "
            "(df['Churn']=='Yes').groupby(df['PaymentMethod']).mean().sort_values(ascending=False) — "
            "this both identifies the top category and gives its rate in one step; never idxmax alone "
            "since the rate itself is also needed. The dataset "
            "identifier column is named customerID (not customer_id). Only use column names and "
            "category values that appear in this schema — "
            "never invent or guess one: "
            f"{self.schema or 'schema unavailable'}. "
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
        for raw_step in raw_steps:
            step = self._normalize_step(raw_step)
            if not isinstance(step, dict) or step.get("tool") not in {"data", "model"}:
                raise PlanError("Plan contains an unknown tool")
            action, arguments = step.get("action"), step.get("arguments", {})
            if not isinstance(action, str) or not isinstance(arguments, dict):
                raise PlanError("Each plan step needs action and arguments")
            steps.append(PlanStep(step["tool"], action, arguments))
        return steps

    @staticmethod
    def _normalize_step(step: Any) -> Any:
        """Coerce the two compact shapes free-tier models tend to emit into
        the ``{"tool", "action", "arguments"}`` contract the executor
        expects, instead of rejecting the whole plan over formatting.

        1. ``{"data/describe": {...}}`` — the tool/action pair is the sole
           key, arguments is the value.
        2. ``{"tool": "data/describe", "arguments": {...}}`` — the slash
           notation from the prompt's own prose (e.g. "model/project")
           leaks directly into the "tool" field instead of being split into
           separate "tool"/"action" keys.
        """
        if not isinstance(step, dict):
            return step
        if "tool" not in step and len(step) == 1:
            name, arguments = next(iter(step.items()))
            if isinstance(name, str) and "/" in name and isinstance(arguments, dict):
                tool, action = name.split("/", 1)
                return {"tool": tool, "action": action, "arguments": arguments}
        tool_value = step.get("tool")
        if isinstance(tool_value, str) and "/" in tool_value and "action" not in step:
            tool, action = tool_value.split("/", 1)
            return {"tool": tool, "action": action, "arguments": step.get("arguments", {})}
        return step
