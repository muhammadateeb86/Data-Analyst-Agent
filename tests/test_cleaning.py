import pandas as pd

from src.model.cleaning import load_and_clean_data, get_feature_target_split, TARGET_COLUMN

RAW_PATH = "data/raw_churn.csv"


def test_total_charges_has_no_nulls_after_cleaning():
    df = load_and_clean_data(RAW_PATH)
    assert df["TotalCharges"].isnull().sum() == 0


def test_total_charges_is_numeric_after_cleaning():
    df = load_and_clean_data(RAW_PATH)
    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])


def test_blank_total_charges_rows_filled_with_zero_not_mean():
    df = load_and_clean_data(RAW_PATH)
    zero_tc_zero_tenure = df[(df["tenure"] == 0)]
    assert (zero_tc_zero_tenure["TotalCharges"] == 0.0).all()


def test_customer_id_is_unique_index():
    df = load_and_clean_data(RAW_PATH)
    assert df.index.is_unique
    assert df.index.name == "customerID"


def test_no_duplicate_customer_ids():
    df = load_and_clean_data(RAW_PATH)
    assert not df.index.duplicated().any()


def test_feature_only_duplicates_are_all_tenure_one():
    """73 rows share an identical feature profile with another row once
    customerID and Churn are excluded (see cleaning.py docstring). Confirm
    this stays explained by tenure==1 rather than silently growing/changing
    with the data."""
    df = load_and_clean_data(RAW_PATH)
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    dup_mask = df.duplicated(subset=feature_cols, keep=False)
    assert dup_mask.sum() == 73
    assert (df.loc[dup_mask, "tenure"] == 1).all()


def test_feature_target_split_shapes_match():
    df = load_and_clean_data(RAW_PATH)
    X, y = get_feature_target_split(df)
    assert len(X) == len(y) == len(df)
    assert TARGET_COLUMN not in X.columns
    assert set(y.unique()) <= {0, 1}


def test_row_count_and_column_count_unchanged_by_cleaning():
    raw = pd.read_csv(RAW_PATH)
    cleaned = load_and_clean_data(RAW_PATH)
    assert len(cleaned) == len(raw)
    # -1 because customerID moved from column to index
    assert len(cleaned.columns) == len(raw.columns) - 1
