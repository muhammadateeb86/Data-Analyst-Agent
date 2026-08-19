"""One-call prose synthesizer constrained to fact placeholders."""

from __future__ import annotations

import json
from typing import Callable

from src.agent.fact_ledger import FactLedger
from src.agent.grounding import ground


class Synthesizer:
    def __init__(self, llm: Callable[[str], str]) -> None:
        self.llm = llm

    def synthesize(self, question: str, ledger: FactLedger) -> str:
        prompt = (
            "Answer the user's question using the supplied facts. Do not write any digits or numerical "
            "values yourself. Insert each required fact exactly as {{fact_N}} and explain it in prose. "
            "Facts: " + json.dumps(ledger.as_prompt_data(), default=str) + " Question: " + question
        )
        return ground(self.llm(prompt), ledger)
