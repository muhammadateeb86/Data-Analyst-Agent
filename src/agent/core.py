"""Small plan-act-verify-synthesize loop with at most one retry."""

from __future__ import annotations

from typing import Callable

from src.agent.executor import Executor
from src.agent.fact_ledger import FactLedger
from src.agent.planner import Planner
from src.agent.synthesizer import Synthesizer
from src.agent.verifier import Verifier
from src.tools.data_tools import DataTools


class DataAnalystAgent:
    """A successful request uses two LLM calls: planning then synthesis.

    A failed verification can use one bounded re-plan; it never loops.
    ``llm`` is intentionally a tiny injectable callable, keeping provider SDKs
    and credentials out of the agent/data core.
    """
    def __init__(self, llm: Callable[[str], str], data_tools: DataTools | None = None, retry_limit: int = 1) -> None:
        self.planner = Planner(llm)
        self.executor = Executor(data_tools or DataTools())
        self.verifier = Verifier()
        self.synthesizer = Synthesizer(llm)
        self.retry_limit = retry_limit

    def answer(self, question: str) -> str:
        feedback = None
        for attempt in range(self.retry_limit + 1):
            ledger = FactLedger()
            steps = self.planner.plan(question, feedback)
            self.executor.execute(steps, ledger)
            valid, feedback = self.verifier.verify(ledger)
            if valid:
                return self.synthesizer.synthesize(question, ledger)
        raise RuntimeError(f"Unable to verify tool results after {self.retry_limit + 1} attempt(s): {feedback}")
