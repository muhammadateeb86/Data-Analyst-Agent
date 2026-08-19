"""Deterministically substitute ledger facts and reject ungrounded numerals."""

from __future__ import annotations

import json
import re

from src.agent.fact_ledger import FactLedger

_PLACEHOLDER = re.compile(r"\{\{(fact_\d+)\}\}")
_NUMERAL = re.compile(r"(?<![A-Za-z_])-?\d+(?:\.\d+)?")


def ground(template: str, ledger: FactLedger) -> str:
    """Replace only explicit fact placeholders; reject all other digit numerals.

    This makes the no-invented-numbers property mechanical rather than a
    prompt-following convention.  Values are serialized from the ledger only.
    """
    if _NUMERAL.search(template):
        raise ValueError("Synthesis contained an ungrounded numerical literal")
    used: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        fact_id = match.group(1)
        used.add(fact_id)
        return json.dumps(ledger.get(fact_id).value, ensure_ascii=False, default=str)

    answer = _PLACEHOLDER.sub(replace, template)
    if "{{" in answer or not used:
        raise ValueError("Synthesis must reference at least one valid fact placeholder")
    return answer
