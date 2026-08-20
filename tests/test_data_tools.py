import pandas as pd
import pytest

from src.tools.data_tools import DataTools, UnsafeQueryError


@pytest.fixture
def tools():
    return DataTools(pd.DataFrame({"segment": ["A", "A", "B"], "risk": [0.2, 0.8, 0.4]}))


def test_safe_query_can_filter_and_aggregate(tools):
    result = tools.query('df[df["risk"] >= 0.4].groupby("segment")["risk"].mean()')
    assert result["values"] == {"A": 0.8, "B": 0.4}


def test_safe_query_rejects_imports_and_private_attributes(tools):
    with pytest.raises(UnsafeQueryError):
        tools.query('__import__("os").system("whoami")')
    with pytest.raises(UnsafeQueryError):
        tools.query("df.__class__")


@pytest.mark.parametrize("expression", [
    # Dunder-chain sandbox escape (reach object -> subclasses -> os/subprocess).
    "().__class__.__bases__[0].__subclasses__()",
    "df.__class__.__mro__",
    "df.__dict__",
    # Indirection around the "only df is a callable target" rule.
    'getattr(df, "to_csv")',
    'globals()',
    'locals()',
    'vars(df)',
    # Bare builtins / arbitrary-code-execution primitives.
    'eval("1+1")',
    'exec("import os")',
    'open("data/cleaned_churn.csv")',
    'compile("1", "<s>", "eval")',
    '__builtins__',
    # Comprehensions and lambdas (arbitrary code inside an expression).
    "[c for c in df.columns]",
    "{c: 1 for c in df.columns}",
    "(lambda: df)()",
    "df.apply(lambda row: row)",
    # Mutating / file-writing / state-changing dataframe methods.
    'df.to_csv("out.csv")',
    'df.to_pickle("out.pkl")',
    "df.assign(new_col=1)",
    "df.drop(columns=['segment'])",
    "df.rename(columns={'segment': 'x'})",
    "df.replace(1, 2)",
    "df.pop('segment')",
    "df.insert(0, 'x', 1)",
    "df.update(df)",
    "df.eval('x = 1')",
    "df.query('risk > 0')",
    # Statements (not expressions) — assignment, imports, multi-statement.
    "df['x'] = 1",
    "import os",
    "df.risk; import os",
    # Starred/unpacking indirection and the walrus operator.
    "df.groupby(*['segment'])",
    "(x := df)",
])
def test_safe_query_rejects_sandbox_escapes(tools, expression):
    with pytest.raises(UnsafeQueryError):
        tools.query(expression)


def test_safe_query_still_allows_legitimate_chained_operations(tools):
    """The adversarial coverage above must not be so broad it breaks real
    usage — multi-step read-only chains (filter, groupby, multi-aggregate,
    sort, head) are exactly what the planner is expected to emit."""
    result = tools.query(
        'df[df["risk"] > 0.1].groupby("segment")["risk"].agg(["mean", "max"]).sort_values("mean").head(5)'
    )
    assert result["type"] == "dataframe"
    assert result["row_count"] == 2


def test_filter_and_groupby_aggregate(tools):
    assert tools.filter({"risk": {"gte": 0.4}})["row_count"] == 2
    result = tools.groupby_aggregate("segment", {"risk": "mean"})
    assert result["row_count"] == 2


def test_count_rows(tools):
    assert tools.count_rows() == {"row_count": 3}
