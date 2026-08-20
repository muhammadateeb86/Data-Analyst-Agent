import pandas as pd

from src.agent.fact_ledger import FactLedger
from src.agent.grounding import ground
from src.tools.data_tools import DataTools


def test_dataframe_fact_renders_as_readable_rows_not_raw_json():
    ledger = FactLedger()
    ledger.add("data", {
        "type": "dataframe",
        "rows": [{"Contract": "Month-to-month", "MonthlyCharges": 66.398490322580}],
        "row_count": 1, "truncated": False,
    }, "data/groupby_aggregate")
    answer = ground("Here is the breakdown: {{fact_1}}.", ledger)
    assert '"type": "dataframe"' not in answer
    assert "Contract: Month-to-month" in answer
    assert "MonthlyCharges: 66.3985" in answer


def test_series_fact_renders_as_key_value_pairs():
    ledger = FactLedger()
    ledger.add("data", {"type": "series", "values": {"A": 0.2, "B": 0.4}, "row_count": 2, "truncated": False},
               "data/query")
    answer = ground("Result: {{fact_1}}.", ledger)
    assert answer == "Result: A: 0.2, B: 0.4."


def test_model_fact_still_renders_as_compact_json():
    ledger = FactLedger()
    ledger.add("model", {"risk_score": 0.42, "source": "customer_id", "customer_id": "abc"}, "model/predict")
    answer = ground("{{fact_1}}", ledger)
    assert '"risk_score": 0.42' in answer
    assert '"customer_id": "abc"' in answer


def test_single_key_dict_fact_renders_as_bare_value():
    """{"row_count": 7043} from data/count_rows should read as `7043`, not
    the raw `{"row_count": 7043}` JSON object, inline in a sentence."""
    ledger = FactLedger()
    ledger.add("data", {"row_count": 7043}, "data/count_rows")
    answer = ground("The dataset contains {{fact_1}} customers.", ledger)
    assert answer == "The dataset contains 7043 customers."


def test_schema_summary_lists_low_cardinality_categories():
    tools = DataTools(pd.DataFrame({"Contract": ["Month-to-month", "Two year"], "tenure": [1, 24]}))
    summary = tools.schema_summary()
    assert "Contract (text): ['Month-to-month', 'Two year']" in summary
    assert "tenure (int64)" in summary
