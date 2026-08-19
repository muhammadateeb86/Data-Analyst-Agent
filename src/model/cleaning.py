"""
Data loading and cleaning for the Telco customer churn dataset.

Cleaning decisions are documented in the training notebook
(notebook/churn_eda_and_training.ipynb). Summary:

- `TotalCharges` is stored as a string with 11 rows containing a single
  blank/whitespace value instead of a number. Investigation showed all 11
  rows have `tenure == 0` (brand-new customers who have not been billed
  yet) and none have churned. `TotalCharges` correlates almost perfectly
  (r ~ 0.9996) with `MonthlyCharges * tenure` for all other rows, so the
  structurally correct value for a tenure=0 customer is 0.0, not a mean
  imputation (which would fabricate billing history that doesn't exist).
- `customerID` is used as the DataFrame index (unique for all 7043 rows),
  not a model feature.
- No duplicate `customerID` values, no malformed IDs, no stray whitespace
  in categorical columns, and the "No phone service" / "No internet
  service" category labels are fully consistent with `PhoneService` /
  `InternetService` (not data errors).
- 73 rows (33 groups) share an identical *feature* profile with at least
  one other row once `customerID` and `Churn` are excluded. All 73 have
  `tenure == 1`: at tenure=1, `TotalCharges` equals `MonthlyCharges`
  exactly, which removes a degree of freedom, and `MonthlyCharges` is
  effectively quantized by plan pricing. These are distinct customers
  (different `customerID`s) who coincidentally share a profile in their
  first month — not duplicate records of the same customer — and are kept
  as-is. Notably, 18 of the 33 groups have MIXED `Churn` outcomes: the
  identical feature profile produced both "Yes" and "No" for different
  real customers. This sets a hard ceiling on achievable model
  performance for those profiles — no classifier can separate them with
  the given features, and this is a real (not fixable) source of the
  gap between the model's ROC-AUC and a hypothetical perfect score.
"""

from __future__ import annotations

import pandas as pd

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]


def load_and_clean_data(path: str) -> pd.DataFrame:
    """Load the raw churn CSV and apply the documented cleaning steps.

    Returns a DataFrame indexed by customerID with TotalCharges coerced
    to numeric and blanks (tenure=0 customers) filled with 0.0.
    """
    df = pd.read_csv(path)
    df = df.set_index(ID_COLUMN)

    total_charges_numeric = pd.to_numeric(df["TotalCharges"], errors="coerce")
    blank_mask = total_charges_numeric.isnull()

    if blank_mask.any():
        # Sanity check the assumption this function's docstring relies on.
        # If a future data refresh introduces blanks NOT tied to tenure=0,
        # fail loudly rather than silently fill them with 0.
        if not (df.loc[blank_mask, "tenure"] == 0).all():
            raise ValueError(
                "Found blank TotalCharges rows with tenure != 0 — "
                "the tenure=0 assumption behind zero-filling no longer "
                "holds and this needs re-investigation before cleaning."
            )
        df["TotalCharges"] = total_charges_numeric.fillna(0.0)
    else:
        df["TotalCharges"] = total_charges_numeric

    return df


def get_feature_target_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned DataFrame into feature matrix X and binary target y."""
    y = (df[TARGET_COLUMN] == "Yes").astype(int)
    X = df.drop(columns=[TARGET_COLUMN])
    return X, y


def get_categorical_features(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in NUMERIC_FEATURES]
