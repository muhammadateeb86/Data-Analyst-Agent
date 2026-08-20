import pandas as pd

from src.agent.executor import Executor
from src.agent.fact_ledger import FactLedger
from src.agent.planner import PlanStep
from src.tools.data_tools import DataTools


def test_projection_uses_verified_customer_features_and_requested_overrides_only(monkeypatch):
    calls = []

    def fake_predict_risk(customer_id=None, features=None):
        calls.append({"customer_id": customer_id, "features": features})
        return {"risk_score": 0.25 if customer_id else 0.1, "top_factors": [],
                "source": "customer_id" if customer_id else "hypothetical", "customer_id": customer_id}

    monkeypatch.setattr("src.agent.executor.predict_risk", fake_predict_risk)
    data = DataTools(pd.DataFrame([{
        "customerID": "C-1", "Churn": "No", "Contract": "Month-to-month", "tenure": 1,
        "MonthlyCharges": 50.0,
    }]))
    executor, ledger = Executor(data), FactLedger()
    executor.execute([
        PlanStep("data", "customer_features", {"customer_id": "C-1"}),
        PlanStep("model", "predict", {"customer_id": "C-1"}),
        PlanStep("model", "project", {"features_from_fact": "fact_1", "overrides": {"Contract": "Two year", "tenure": 24}}),
    ], ledger)

    assert ledger.get("fact_1").value == {"customer_id": "C-1", "features": {
        "Contract": "Month-to-month", "tenure": 1, "MonthlyCharges": 50.0,
    }}
    assert calls[0] == {"customer_id": "C-1", "features": None}
    assert calls[1] == {"customer_id": None, "features": {
        "Contract": "Two year", "tenure": 24, "MonthlyCharges": 50.0,
    }}
    assert ledger.get("fact_3").value["base_feature_fact"] == "fact_1"
    assert ledger.get("fact_3").value["risk_score"] == 0.1
