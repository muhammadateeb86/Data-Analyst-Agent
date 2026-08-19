"""Safe, read-only analysis tools for the cleaned churn dataframe.

The expression interface is deliberately small.  It is useful for an LLM
planner, but is AST-validated before evaluation and has no imports, builtins,
file access, assignment, or mutation methods available.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_DATA_PATH = Path("data/cleaned_churn.csv")
_METHODS = {
    "agg", "aggregate", "count", "describe", "groupby", "head", "max", "mean",
    "median", "min", "nunique", "reset_index", "size", "sort_index", "sort_values",
    "sum", "tail", "to_frame", "value_counts", "isin",
}
_AGGREGATIONS = {"count", "max", "mean", "median", "min", "nunique", "size", "sum"}
_ALLOWED_NODES = (
    ast.Expression, ast.Name, ast.Load, ast.Attribute, ast.Subscript, ast.Slice,
    ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.keyword, ast.Call,
    ast.Compare, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Eq, ast.NotEq, ast.Gt,
    ast.GtE, ast.Lt, ast.LtE, ast.In, ast.NotIn, ast.And, ast.Or, ast.BitAnd,
    ast.BitOr, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd,
)


class UnsafeQueryError(ValueError):
    """Raised when an expression is outside the read-only query language."""


class _QueryValidator(ast.NodeVisitor):
    def generic_visit(self, node: ast.AST) -> None:
        if not isinstance(node, _ALLOWED_NODES):
            raise UnsafeQueryError(f"Unsupported syntax: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id != "df":
            raise UnsafeQueryError("Only the dataframe name 'df' is available")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_") or "__" in node.attr:
            raise UnsafeQueryError("Private attributes are not allowed")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Attribute) or node.func.attr not in _METHODS:
            raise UnsafeQueryError("Only approved dataframe operations may be called")
        if any(keyword.arg is None for keyword in node.keywords):
            raise UnsafeQueryError("Starred keyword arguments are not allowed")
        self.generic_visit(node)


def _json_value(value: Any, max_rows: int = 100) -> Any:
    """Convert pandas/numpy output to a bounded JSON-compatible value."""
    if isinstance(value, pd.DataFrame):
        frame = value.head(max_rows).replace({np.nan: None})
        frame.columns = [str(column) for column in frame.columns]
        return {"type": "dataframe", "rows": frame.to_dict(orient="records"),
                "row_count": int(len(value)), "truncated": len(value) > max_rows}
    if isinstance(value, pd.Series):
        series = value.head(max_rows).replace({np.nan: None})
        return {"type": "series", "values": {str(key): item for key, item in series.to_dict().items()}, "row_count": int(len(value)),
                "truncated": len(value) > max_rows}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


class DataTools:
    """Read-only EDA operations over a dataframe.

    ``query`` is for planner-generated expressions; the named methods are a
    clearer API for application code and tests.
    """

    def __init__(self, dataframe: pd.DataFrame | None = None, data_path: str | Path = DEFAULT_DATA_PATH):
        self.df = dataframe.copy(deep=True) if dataframe is not None else pd.read_csv(data_path)

    @property
    def columns(self) -> list[str]:
        return self.df.columns.tolist()

    def query(self, expression: str) -> Any:
        if not isinstance(expression, str) or not expression.strip():
            raise UnsafeQueryError("Query expression must be a non-empty string")
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise UnsafeQueryError("Invalid query syntax") from exc
        _QueryValidator().visit(tree)
        # Validator guarantees the namespace contains no callable names or
        # builtins.  The dataframe is a private copy and only non-mutating
        # methods are permitted.
        result = eval(compile(tree, "<safe-data-query>", "eval"), {"__builtins__": {}}, {"df": self.df})
        return _json_value(result)

    def describe(self, columns: Iterable[str] | None = None) -> dict[str, Any]:
        selected = list(columns) if columns is not None else self.columns
        self._require_columns(selected)
        return _json_value(self.df[selected].describe(include="all").transpose().reset_index(names="column"))

    def filter(self, filters: dict[str, Any]) -> dict[str, Any]:
        """Filter exact values, or comparison dicts such as ``{"tenure": {"gte": 12}}``."""
        result = self.df
        self._require_columns(filters)
        valid_ops = {"eq", "ne", "gt", "gte", "lt", "lte", "in"}
        for column, condition in filters.items():
            if not isinstance(condition, dict):
                result = result[result[column] == condition]
                continue
            if set(condition) - valid_ops or len(condition) != 1:
                raise ValueError(f"Unsupported filter for {column!r}")
            op, value = next(iter(condition.items()))
            series = result[column]
            masks = {"eq": series == value, "ne": series != value, "gt": series > value,
                     "gte": series >= value, "lt": series < value, "lte": series <= value,
                     "in": series.isin(value if isinstance(value, list) else [value])}
            result = result[masks[op]]
        return _json_value(result)

    def groupby_aggregate(self, by: str | list[str], aggregations: dict[str, str | list[str]]) -> dict[str, Any]:
        groups = [by] if isinstance(by, str) else list(by)
        self._require_columns(groups)
        self._require_columns(aggregations)
        invalid = {str(op) for ops in aggregations.values() for op in (ops if isinstance(ops, list) else [ops])} - _AGGREGATIONS
        if invalid:
            raise ValueError(f"Unsupported aggregation(s): {sorted(invalid)}")
        result = self.df.groupby(groups, dropna=False).agg(aggregations).reset_index()
        return _json_value(result)

    def _require_columns(self, columns: Iterable[str]) -> None:
        unknown = set(columns) - set(self.df.columns)
        if unknown:
            raise ValueError(f"Unknown dataframe column(s): {sorted(unknown)}")


def execute_dataframe_query(expression: str, dataframe: pd.DataFrame | None = None) -> Any:
    """Convenience wrapper used by a tool-calling executor."""
    return DataTools(dataframe=dataframe).query(expression)
