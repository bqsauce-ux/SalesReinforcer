
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score



def build_pipeline(X):

    X = X.copy()

    # Identify categorical columns.
    categorical_columns = X.select_dtypes(
        exclude=np.number
    ).columns

    # Convert categorical values to consistent strings.
    for col in categorical_columns:
        X[col] = X[col].map(
            lambda value: (
                str(value)
                if pd.notna(value)
                else np.nan
            )
        )

    # Identify numeric and categorical features.
    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    # Numeric preprocessing.
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    # Categorical preprocessing.
    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__MISSING__"
                )
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    # Combine preprocessing.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Classification model.
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    return pipeline


def calculate_reward(
    X,
    y,
    selected_features,
    complexity_penalty=0.05,
    cv_splits=3,
):

    if len(selected_features) == 0:
        return -1.0

    X_selected = X[selected_features].reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    if y.isna().any() or y.nunique() != 2:
        return -1.0

    min_class_count = int(y.value_counts().min())

    if min_class_count < 2:
        return -1.0

    n_splits = min(cv_splits, min_class_count)

    pipeline = build_pipeline(X_selected)

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    try:
        scores = cross_val_score(
            pipeline,
            X_selected,
            y,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        )

        auc = float(np.mean(scores))

        penalty = (
            complexity_penalty
            * len(selected_features)
            / X.shape[1]
        )

        return auc - penalty

    except (ValueError, TypeError):
        return -1.0


def subscription_reward(y_true, y_pred):

    rewards = np.where(
        np.asarray(y_pred) == np.asarray(y_true),
        10,
        -5,
    )

    return float(np.mean(rewards))


def call_reward(call_occurred):

    if pd.isna(call_occurred):
        return None

    return 3.0 if call_occurred else -1.0
