import json

import pandas as pd
import pytest

from src.agent import DataAnalystAgent, FactLedger, ground
from src.agent.planner import Planner
from src.tools.data_tools import DataTools


class FakeLLM:
    def __init__(self):
        self.calls = []

    def __call__(self, prompt):
        self.calls.append(prompt)
        if "Return JSON only" in prompt:
            return json.dumps({"steps": [{"tool": "data", "action": "query", "arguments": {"expression": 'df["risk"].mean()'}}]})
        return "The calculated mean risk is {{fact_1}}."


def test_agent_uses_two_calls_and_grounds_tool_output():
    llm = FakeLLM()
    agent = DataAnalystAgent(llm, DataTools(pd.DataFrame({"risk": [0.2, 0.4]})))
    answer = agent.answer("What is the average risk?")
    # Numbers are grounded from the raw tool value but rounded for
    # readability (see src/agent/grounding.py:_format_number) rather than
    # echoing full floating-point noise like 0.30000000000000004.
    assert answer == "The calculated mean risk is 0.3."
    assert len(llm.calls) == 2


def test_grounding_rejects_numbers_not_from_fact_ledger():
    ledger = FactLedger()
    ledger.add("data", 0.4, "data/query")
    with pytest.raises(ValueError):
        ground("The result is 40 percent and {{fact_1}}", ledger)


def test_grounding_accepts_double_digit_fact_ids():
    """Regression: {{fact_10}}'s trailing "0" is preceded by "1", not a
    letter/underscore, so scanning the raw template for stray numerals used
    to false-positive on any fact ID of 10 or higher, breaking synthesis for
    every ledger with 10+ facts (a wide describe(), a groupby over several
    categories, several batch predictions)."""
    ledger = FactLedger()
    for i in range(11):
        ledger.add("data", i, f"data/step{i}")
    answer = ground("Result: {{fact_11}}", ledger)
    assert answer == "Result: 10"
    # The fix must not reopen the door to a genuinely stray numeral sitting
    # right next to a valid multi-digit placeholder.
    with pytest.raises(ValueError):
        ground("Value 99 and {{fact_11}}", ledger)


def test_planner_accepts_compact_tool_action_steps():
    planner = Planner(lambda _: json.dumps({"steps": [
        {"data/describe": {}},
        {"data/groupby_aggregate": {"by": "Contract", "aggregations": {"MonthlyCharges": "mean"}}},
    ]}))
    steps = planner.plan("Summarize charges by contract")
    assert [(step.tool, step.action) for step in steps] == [
        ("data", "describe"), ("data", "groupby_aggregate"),
    ]


def test_planner_accepts_slash_in_tool_field_with_separate_arguments_key():
    """Regression: the LLM sometimes echoes the prompt's own 'model/project'
    notation straight into the "tool" field alongside a separate
    "arguments" key, e.g. {"tool": "model/project", "arguments": {...}},
    instead of splitting it into {"tool": "model", "action": "project"}.
    This used to be rejected outright as 'Plan contains an unknown tool'."""
    planner = Planner(lambda _: json.dumps({"steps": [
        {"tool": "data/customer_features", "arguments": {"customer_id": "7590-VHVEG"}},
        {"tool": "model/predict", "arguments": {"customer_id": "7590-VHVEG"}},
        {"tool": "model/project", "arguments": {"features_from_fact": "fact_1", "overrides": {"Contract": "Two year", "tenure": 24}}},
    ]}))
    steps = planner.plan("Compare current vs projected risk")
    assert [(step.tool, step.action) for step in steps] == [
        ("data", "customer_features"), ("model", "predict"), ("model", "project"),
    ]
    assert steps[2].arguments == {"features_from_fact": "fact_1", "overrides": {"Contract": "Two year", "tenure": 24}}


def test_planner_prompt_directs_customer_risk_to_model_tool():
    captured = []
    planner = Planner(lambda prompt: captured.append(prompt) or json.dumps({"steps": [
        {"model/predict": {"customer_id": "7590-VHVEG"}},
    ]}))
    steps = planner.plan("What is the current churn risk for customer 7590-VHVEG?")
    assert steps[0].tool == "model"
    assert steps[0].action == "predict"
    assert "customerID (not customer_id)" in captured[0]


def test_planner_prompt_forbids_partial_features_for_named_customer_hypothetical():
    captured = []
    planner = Planner(lambda prompt: captured.append(prompt) or json.dumps({"steps": [
        {"tool": "data", "action": "customer_features", "arguments": {"customer_id": "X"}},
        {"tool": "model", "action": "predict", "arguments": {"customer_id": "X"}},
        {"tool": "model", "action": "project", "arguments": {"features_from_fact": "fact_1", "overrides": {"Contract": "Two year"}}},
    ]}))
    planner.plan("What would X's churn risk be if they switched to a Two year contract?")
    assert "Never call model/predict directly with a partial" in captured[0]


def test_planner_prompt_directs_rate_by_category_to_grouped_query():
    captured = []
    planner = Planner(lambda prompt: captured.append(prompt) or json.dumps({"steps": [
        {"tool": "data", "action": "query", "arguments": {
            "expression": "(df['Churn']=='Yes').groupby(df['PaymentMethod']).mean().sort_values(ascending=False)"}},
    ]}))
    planner.plan("Which payment method has the highest churn rate?")
    assert "groupby(df['PaymentMethod'])" in captured[0]


def test_answer_raises_runtime_error_when_plan_retries_exhausted():
    """Regression: previously the final-attempt PlanError propagated as a
    bare PlanError (a ValueError) instead of the same RuntimeError contract
    used when verification retries are exhausted, making failure handling
    inconsistent depending on which stage failed last."""
    from src.agent.core import DataAnalystAgent

    def always_bad_plan(prompt):
        return "not valid json"

    agent = DataAnalystAgent(always_bad_plan, DataTools(pd.DataFrame({"Churn": ["Yes"]})))
    with pytest.raises(RuntimeError, match="Unable to produce a valid plan"):
        agent.answer("anything")


def test_customer_risk_uses_model_prediction_locally():
    calls = []

    def llm(prompt):
        calls.append(prompt)
        if "Return JSON only" in prompt:
            return json.dumps({"steps": [{"model/predict": {"customer_id": "7590-VHVEG"}}]})
        return "The verified customer prediction is {{fact_1}}."

    answer = DataAnalystAgent(llm).answer("What is the current churn risk for customer 7590-VHVEG?")
    assert '"source": "customer_id"' in answer
    assert '"customer_id": "7590-VHVEG"' in answer
    assert len(calls) == 2


def test_customer_risk_replaces_synthesized_literal_with_model_fact():
    calls = []

    def llm(prompt):
        calls.append(prompt)
        if "Return JSON only" in prompt:
            return json.dumps({"steps": [{"model/predict": {"customer_id": "7590-VHVEG"}}]})
        return "The customer's churn risk is 0.42."

    answer = DataAnalystAgent(llm).answer("What is the current churn risk for customer 7590-VHVEG?")
    assert "0.42" not in answer
    assert answer.startswith("Verified tool result: ")
    assert '"source": "customer_id"' in answer
    assert '"risk_score":' in answer
    assert len(calls) == 2
