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
            "values yourself. Every numerical value must be represented by its exact {{fact_N}} "
            "placeholder and substituted later. For a prediction, reference the model fact containing "
            "the risk score; never reproduce that score. Insert each required fact exactly as {{fact_N}} "
            "and explain it in prose. Reference each fact placeholder at most once — a fact already "
            "shown fully answers every part of the question it covers, so do not repeat the same "
            "placeholder again to re-explain it row by row. Keep the answer concise. "
            "Facts: " + json.dumps(ledger.as_prompt_data(), default=str) + " Question: " + question
        )
        draft = self.llm(prompt)
        try:
            return ground(draft, ledger)
        except ValueError:
            # A draft containing a literal number (or an empty/unresolved
            # placeholder) is never shown. Fall back to a deterministic
            # template built straight from the fact ledger instead.
            #
            # Prefer model facts when any exist: the common shape here is
            # customer_features (data) -> predict (model) -> project (model),
            # where customer_features is scaffolding for the projection, not
            # something the user asked to see directly — the risk score(s)
            # are the answer. Falling back to *only* the model facts avoids
            # dumping a customer's full ~17-feature raw dict into the answer.
            # When there are no model facts at all (pure EDA questions),
            # every fact is shown instead.
            model_facts = [fact.id for fact in ledger.facts if fact.tool == "model"]
            fact_ids = model_facts or [fact.id for fact in ledger.facts]
            template = "Verified tool result: " + " and ".join(f"{{{{{fact_id}}}}}" for fact_id in fact_ids)
            return ground(template, ledger)
