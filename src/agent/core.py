"""Small plan-act-verify-synthesize loop with at most one retry."""

from __future__ import annotations

from typing import Callable

from src.agent.executor import Executor
from src.agent.fact_ledger import FactLedger
from src.agent.planner import PlanError, Planner
from src.agent.synthesizer import Synthesizer
from src.agent.verifier import Verifier
from src.tools.data_tools import DataTools


class DataAnalystAgent:
    """A successful request uses two LLM calls: planning then synthesis.

    A failed plan or verification triggers a re-plan with the specific
    failure fed back into the prompt (a malformed tool call, a missing
    argument, an empty result). ``retry_limit`` bounds how many of these
    re-plans are allowed (default 2, so up to 3 total planning attempts)
    before the agent gives up and raises instead of looping forever.
    ``llm`` is intentionally a tiny injectable callable, keeping provider
    SDKs and credentials out of the agent/data core.
    """
    def __init__(self, llm: Callable[[str], str], data_tools: DataTools | None = None, retry_limit: int = 2) -> None:
        data_tools = data_tools or DataTools()
        self.planner = Planner(llm, schema=data_tools.schema_summary())
        self.executor = Executor(data_tools)
        self.verifier = Verifier()
        self.synthesizer = Synthesizer(llm)
        self.retry_limit = retry_limit

    def answer(self, question: str) -> str:
        feedback = None
        for attempt in range(self.retry_limit + 1):
            ledger = FactLedger()
            try:
                steps = self.planner.plan(question, feedback)
            except PlanError as exc:
                feedback = str(exc)
                if attempt >= self.retry_limit:
                    raise RuntimeError(
                        f"Unable to produce a valid plan after {self.retry_limit + 1} attempt(s): {feedback}"
                    ) from exc
                continue
            self.executor.execute(steps, ledger)
            valid, feedback = self.verifier.verify(ledger)
            if valid:
                return self.synthesizer.synthesize(question, ledger)
        raise RuntimeError(f"Unable to verify tool results after {self.retry_limit + 1} attempt(s): {feedback}")
