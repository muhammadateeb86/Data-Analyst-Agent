import pandas as pd
import pytest

from src.tools.data_tools import DataTools
from src.tools.segment_tool import segment_churn_risk


@pytest.fixture(scope="module")
def data_tools():
    return DataTools(data_path="data/cleaned_churn.csv")


def test_segment_risk_groups_and_sorts_by_avg_risk(data_tools):
    result = segment_churn_risk(data_tools, by="Contract")
    assert result["type"] == "dataframe"
    contracts = [row["Contract"] for row in result["rows"]]
    assert set(contracts) == {"Month-to-month", "One year", "Two year"}
    risks = [row["avg_predicted_risk"] for row in result["rows"]]
    assert risks == sorted(risks, reverse=True)
    total_customers = sum(row["customer_count"] for row in result["rows"])
    assert total_customers == len(data_tools.df)


def test_segment_risk_respects_filters(data_tools):
    result = segment_churn_risk(data_tools, by="Contract", filters={"InternetService": "Fiber optic"})
    total_customers = sum(row["customer_count"] for row in result["rows"])
    expected = int((data_tools.df["InternetService"] == "Fiber optic").sum())
    assert total_customers == expected


def test_segment_risk_rejects_unknown_column(data_tools):
    with pytest.raises(ValueError):
        segment_churn_risk(data_tools, by="not_a_real_column")


def test_segment_risk_empty_filter_result():
    tools = DataTools(pd.DataFrame({
        "Contract": ["Month-to-month"], "tenure": [1], "MonthlyCharges": [50.0],
        "TotalCharges": [50.0], "SeniorCitizen": [0], "customerID": ["x"], "Churn": ["No"],
    }))
    result = segment_churn_risk(tools, by="Contract", filters={"tenure": {"gte": 999}})
    assert result["row_count"] == 0
