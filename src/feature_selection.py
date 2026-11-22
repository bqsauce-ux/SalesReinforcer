
# src/feature_selection.py

import pandas as pd


class FeatureSelector:
    """
    Prepares CRM features for reinforcement-learning
    feature selection.

    Responsibilities:
    - Separate input features from the target.
    - Remove identifier and leakage columns.
    - Validate the dataset.
    - Return X and y for the RL environment.
    """

    def __init__(
        self,
        df,
        target="Subscribed",
        exclude_columns=None,
    ):

        self.df = df.copy()
        self.target = target

        # Columns that should not be used as predictors.
        self.exclude_columns = exclude_columns or [
            "Id",
        ]

        self.X = None
        self.y = None
        self.feature_names = []

    def prepare_data(self):

        data = self.df.copy()
    
        # Convert subscription dates into a binary target.
        raw_target = data[self.target]
    
        subscription_dates = pd.to_datetime(
            raw_target,
            errors="coerce"
        )
    
        # Valid subscription date = 1
        # Missing or invalid date = 0
        self.y = subscription_dates.notna().astype(int)
    
        # Remove target and excluded columns from features.
        self.X = data.drop(
            columns=[self.target] + self.exclude_columns,
            errors="ignore"
        )
        
        # Normalize date columns.
        self.X = self.normalize_date_columns(self.X)
    
        # Remove columns that contain no information.
        self.X = self.X.dropna(axis=1, how="all")
    
        # Remove constant columns.
        self.X = self.X.loc[
            :, self.X.nunique(dropna=False) > 1
        ]
    
        # Reset indices without dropping records.
        self.X = self.X.reset_index(drop=True)
        self.y = self.y.reset_index(drop=True)
    
        self.feature_names = self.X.columns.tolist()
    
        print("Dataset shape:", self.X.shape)
        print("\nTarget distribution:")
        print(self.y.value_counts())
    
        return self.X, self.y
    
    def normalize_date_columns(self, X):

        X = X.copy()
    
        date_columns = [
            "First Contact",
            "Last Contact",
            "First Call",
            "Signed up for a demo",
            "Filled in customer survey",
            "Did sign up to the platform",
            "Account Manager assigned"
        ]
    
        for col in date_columns:
    
            if col not in X.columns:
                continue
    
            # Normalize strings, datetime objects, and Timestamps.
            dates = pd.to_datetime(
                X[col],
                errors="coerce"
            )
    
            # Convert dates into numeric days since Unix epoch.
            X[col] = (
                dates - pd.Timestamp("1970-01-01")
            ).dt.total_seconds() / 86400
    
            # Preserve missing values as NaN.
            X[col] = X[col].astype(float)
    
        return X

    def get_features(self):

        if self.X is None:
            raise RuntimeError(
                "Call prepare_data() first."
            )

        return self.X.copy()

    def get_target(self):

        if self.y is None:
            raise RuntimeError(
                "Call prepare_data() first."
            )

        return self.y.copy()

    def get_feature_names(self):

        if self.X is None:
            raise RuntimeError(
                "Call prepare_data() first."
            )

        return self.feature_names.copy()

    def get_feature_count(self):

        if self.X is None:
            raise RuntimeError(
                "Call prepare_data() first."
            )

        return len(self.feature_names)
