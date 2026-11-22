
# src/data_preprocessing.py

from pathlib import Path
import numpy as np
import pandas as pd


DATE_COLUMNS = [
    "first_contact",
    "last_contact",
    "first_call",
    "signed_up_for_a_demo",
    "filled_in_customer_survey",
    "did_sign_up_to_the_platform",
    "account_manager_assigned",
    "subscribed",
]


def normalize_columns(df):
    """Convert column names to lowercase snake_case."""

    df = df.copy()

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    return df


def load_data(file_path):
    """Load CRM data from Excel."""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    df = pd.read_excel(file_path, dtype=object)

    return normalize_columns(df)


def clean_data(df):
    """Clean text and parse dates."""

    df = df.copy()

    for col in df.select_dtypes(include="object").columns:

        df[col] = df[col].map(
            lambda x: x.strip() if isinstance(x, str) else x
        )

        df[col] = df[col].replace(
            r"^\s*$", np.nan, regex=True
        )

    # Preserve invalid and missing date diagnostics.
    df.attrs["date_quality"] = {}

    for col in DATE_COLUMNS:

        if col in df.columns:

            original = df[col]

            parsed = pd.to_datetime(
                original,
                errors="coerce"
            )

            df.attrs["date_quality"][col] = {
                "missing": int(original.isna().sum()),
                "invalid": int(
                    (original.notna() & parsed.isna()).sum()
                )
            }

            df[col] = parsed

    df = df.reset_index(drop=True)

    df["source_row_number"] = np.arange(
        1, len(df) + 1
    )

    return df


def engineer_features(df):
    """Create customer and funnel features."""

    df = df.copy()

    if "first_contact" in df.columns:

        df["first_contact_year"] = (
            df["first_contact"].dt.year
        )

        df["first_contact_month"] = (
            df["first_contact"].dt.month
        )

        df["first_contact_dayofweek"] = (
            df["first_contact"].dt.dayofweek
        )

    milestone_columns = [
        "first_call",
        "signed_up_for_a_demo",
        "filled_in_customer_survey",
        "did_sign_up_to_the_platform",
        "account_manager_assigned",
        "subscribed",
    ]

    for col in milestone_columns:

        if col in df.columns:

            df[f"has_{col}"] = (
                df[col].notna().astype(int)
            )

    return df


def prepare_data(
    file_path,
    target="subscribed",
    prediction_stage="initial"
):
    """
    Prepare CRM data for machine learning.

    Unknown subscription outcomes remain nullable.

    prediction_stage:
        initial: Features available before outreach.
        dynamic: Requires a separately defined
                 pipeline-stage feature policy.
    """

    df = load_data(file_path)

    df = clean_data(df)

    df = engineer_features(df)

    if target not in df.columns:
        raise ValueError(
            f"Target '{target}' not found."
        )

    # Preserve unknown outcomes.
    y = pd.Series(
        pd.NA,
        index=df.index,
        dtype="Int64"
    )

    valid_target = df[target].notna()

    y.loc[valid_target] = 1

    # For a confirmed non-subscription outcome,
    # a separate reliable outcome indicator is required.
    # Do not automatically assign zero to missing dates.

    excluded_columns = [
        "id",
        "source_row_number",
        "subscribed",
        "first_call",
        "last_contact",
        "signed_up_for_a_demo",
        "filled_in_customer_survey",
        "did_sign_up_to_the_platform",
        "account_manager_assigned",
        "has_first_call",
        "has_signed_up_for_a_demo",
        "has_filled_in_customer_survey",
        "has_did_sign_up_to_the_platform",
        "has_account_manager_assigned",
        "has_subscribed",
    ]

    if prediction_stage == "initial":

        excluded_columns += [
            "status",
            "stage",
        ]

    elif prediction_stage != "dynamic":

        raise ValueError(
            "prediction_stage must be initial or dynamic"
        )

    feature_columns = [
        col for col in df.columns
        if col not in excluded_columns
    ]

    X = df[feature_columns].copy()

    datetime_cols = X.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns

    X = X.drop(columns=datetime_cols)

    return X, y, df


if __name__ == "__main__":

    X, y, df = prepare_data(
        "data/raw/SalesCRM.xlsx"
    )

    print("Dataset shape:", df.shape)
    print("Feature shape:", X.shape)

    print("\nPositive subscription records:")
    print(int((y == 1).sum()))

    print("\nUnknown outcomes:")
    print(int(y.isna().sum()))

    print("\nFeature columns:")
    print(X.columns.tolist())

    print("\nDate quality:")
    print(df.attrs.get("date_quality"))
