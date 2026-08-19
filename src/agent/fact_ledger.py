"""Immutable record of values that may appear in an agent response."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Fact:
    id: str
    tool: str
    value: Any
    source: str


class FactLedger:
    def __init__(self) -> None:
        self._facts: list[Fact] = []

    def add(self, tool: str, value: Any, source: str) -> Fact:
        fact = Fact(id=f"fact_{len(self._facts) + 1}", tool=tool, value=value, source=source)
        self._facts.append(fact)
        return fact

    def get(self, fact_id: str) -> Fact:
        for fact in self._facts:
            if fact.id == fact_id:
                return fact
        raise KeyError(f"Unknown fact ID: {fact_id}")

    def as_prompt_data(self) -> list[dict[str, Any]]:
        return [{"id": fact.id, "tool": fact.tool, "value": fact.value, "source": fact.source} for fact in self._facts]

    @property
    def facts(self) -> tuple[Fact, ...]:
        return tuple(self._facts)
