import pandas as pd
import pytest

from src.model.predict import predict_churn_risk, PredictionInputError

CLEANED_PATH = "data/cleaned_churn.csv"


@pytest.fixture(scope="module")
def cleaned_df():
    return pd.read_csv(CLEANED_PATH)


@pytest.fixture(scope="module")
def sample_customer_id(cleaned_df):
    return cleaned_df.iloc[0]["customerID"]


@pytest.fixture(scope="module")
def sample_features(cleaned_df):
    row = cleaned_df.iloc[0]
    feature_cols = [c for c in cleaned_df.columns if c not in ("customerID", "Churn")]
    return {c: row[c] for c in feature_cols}


def test_predict_by_customer_id_returns_valid_schema(sample_customer_id):
    result = predict_churn_risk(customer_id=sample_customer_id)
    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["source"] == "customer_id"
    assert result["customer_id"] == sample_customer_id
    assert isinstance(result["top_factors"], list)
    assert len(result["top_factors"]) > 0
    for factor in result["top_factors"]:
        assert factor["direction"] in ("increases_risk", "decreases_risk")


def test_predict_hypothetical_returns_valid_schema(sample_features):
    result = predict_churn_risk(features=sample_features)
    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["source"] == "hypothetical"
    assert result["customer_id"] is None


def test_two_year_contract_lower_risk_than_month_to_month(sample_features):
    """Sanity check the model direction matches the EDA finding that
    long-term contracts reduce churn risk."""
    monthly = dict(sample_features, Contract="Month-to-month", tenure=1)
    two_year = dict(sample_features, Contract="Two year", tenure=24)

    risk_monthly = predict_churn_risk(features=monthly)["risk_score"]
    risk_two_year = predict_churn_risk(features=two_year)["risk_score"]

    assert risk_two_year < risk_monthly


def test_unknown_customer_id_raises():
    with pytest.raises(PredictionInputError):
        predict_churn_risk(customer_id="NOT-AREAL-ID")


def test_no_arguments_raises():
    with pytest.raises(PredictionInputError):
        predict_churn_risk()


def test_both_arguments_raises(sample_customer_id, sample_features):
    with pytest.raises(PredictionInputError):
        predict_churn_risk(customer_id=sample_customer_id, features=sample_features)


def test_missing_feature_raises(sample_features):
    incomplete = dict(sample_features)
    del incomplete["Contract"]
    with pytest.raises(PredictionInputError):
        predict_churn_risk(features=incomplete)


def test_unknown_feature_raises(sample_features):
    bogus = dict(sample_features)
    bogus["NotARealColumn"] = 1
    with pytest.raises(PredictionInputError):
        predict_churn_risk(features=bogus)
