"""Deterministic checks that decide whether a plan needs its one retry."""

from __future__ import annotations

from src.agent.fact_ledger import FactLedger


class Verifier:
    def verify(self, ledger: FactLedger) -> tuple[bool, str | None]:
        if not ledger.facts:
            return False, "No tool results were produced"
        for fact in ledger.facts:
            value = fact.value
            if isinstance(value, dict) and "error" in value:
                return False, f"{fact.source} failed: {value['error']}"
            if isinstance(value, dict) and value.get("row_count") == 0:
                return False, f"{fact.source} returned no rows"
        return True, None
