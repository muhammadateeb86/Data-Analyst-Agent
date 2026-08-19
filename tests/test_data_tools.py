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


def test_filter_and_groupby_aggregate(tools):
    assert tools.filter({"risk": {"gte": 0.4}})["row_count"] == 2
    result = tools.groupby_aggregate("segment", {"risk": "mean"})
    assert result["row_count"] == 2
