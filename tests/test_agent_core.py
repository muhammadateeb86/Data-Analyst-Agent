import json

import pandas as pd
import pytest

from src.agent import DataAnalystAgent, FactLedger, ground
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
    assert answer == "The calculated mean risk is 0.30000000000000004."
    assert len(llm.calls) == 2


def test_grounding_rejects_numbers_not_from_fact_ledger():
    ledger = FactLedger()
    ledger.add("data", 0.4, "data/query")
    with pytest.raises(ValueError):
        ground("The result is 40 percent and {{fact_1}}", ledger)
