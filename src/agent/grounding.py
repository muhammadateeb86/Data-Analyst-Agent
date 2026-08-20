"""Deterministically substitute ledger facts and reject ungrounded numerals."""

from __future__ import annotations

import json
import re
from typing import Any

from src.agent.fact_ledger import FactLedger

_PLACEHOLDER = re.compile(r"\{\{(fact_\d+)\}\}")
_NUMERAL = re.compile(r"(?<![A-Za-z_])-?\d+(?:\.\d+)?")


def _format_number(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, 4)
    return value


def _format_row(row: dict[str, Any]) -> str:
    return ", ".join(f"{key}: {_format_number(value)}" for key, value in row.items())


def render_fact(value: Any) -> str:
    """Render a tool-result value as plain text instead of raw JSON.

    Scalars/short dicts (e.g. a single model prediction) still come through
    as compact JSON, which stays readable at that size. The thing worth
    fixing is dataframe/series-shaped results (``{"type": "dataframe", ...}``
    from ``DataTools``) — those were previously dumped as one giant raw JSON
    blob in the middle of a sentence. Here they become a short, comma
    separated list of rows instead, still built only from ledger data.
    """
    if isinstance(value, dict) and value.get("type") == "dataframe":
        rows = value.get("rows", [])
        if not rows:
            return "no matching rows"
        lines = [_format_row(row) for row in rows]
        text = "; ".join(f"({line})" for line in lines)
        if value.get("truncated"):
            text += f" (showing {len(rows)} of {value.get('row_count', len(rows))} rows)"
        return text
    if isinstance(value, dict) and value.get("type") == "series":
        values = value.get("values", {})
        return ", ".join(f"{key}: {_format_number(val)}" for key, val in values.items()) or "no values"
    if isinstance(value, dict) and len(value) == 1:
        # A single-key result (e.g. {"row_count": 7043} from data/count_rows)
        # reads far better inlined as a bare value than as a JSON object —
        # "contains 7043 customers" instead of `contains {"row_count": 7043}`.
        only_value = next(iter(value.values()))
        if isinstance(only_value, (int, float)) and not isinstance(only_value, bool):
            return json.dumps(_format_number(only_value))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return json.dumps(_format_number(value))
    return json.dumps(value, ensure_ascii=False, default=str)


def ground(template: str, ledger: FactLedger) -> str:
    """Replace only explicit fact placeholders; reject all other digit numerals.

    This makes the no-invented-numbers property mechanical rather than a
    prompt-following convention.  Values are serialized from the ledger only.
    """
    # Scan for stray numerals only *outside* well-formed {{fact_N}} spans.
    # Scanning the raw template directly is wrong: _NUMERAL's lookbehind only
    # blocks a match starting immediately after a letter/underscore, so for
    # a two-digit-or-longer ID like {{fact_10}} it correctly skips the "1"
    # (preceded by "_") but then matches the trailing "0" on its own
    # (preceded by "1", not a letter/underscore) — a false positive that
    # rejects any answer needing a 10th+ fact. Stripping placeholders first
    # removes their digits from consideration entirely, so this can't fire
    # on a valid ID while still catching every genuinely stray numeral
    # (including a malformed near-placeholder, which is left untouched here
    # and still rejected — either by this check or by the trailing "{{" check
    # below, since it's never substituted).
    if _NUMERAL.search(_PLACEHOLDER.sub("", template)):
        raise ValueError("Synthesis contained an ungrounded numerical literal")
    used: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        fact_id = match.group(1)
        used.add(fact_id)
        return render_fact(ledger.get(fact_id).value)

    answer = _PLACEHOLDER.sub(replace, template)
    if "{{" in answer or not used:
        raise ValueError("Synthesis must reference at least one valid fact placeholder")
    return answer
