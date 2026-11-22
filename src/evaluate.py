
# src/evaluate.py

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
)

from reward_function import build_pipeline


class ModelEvaluator:

    def __init__(
        self,
        X,
        y,
        test_size=0.2,
        random_state=42,
    ):

        self.X = X.reset_index(drop=True)
        self.y = pd.Series(y).reset_index(drop=True)

        self.X_train, self.X_test, self.y_train, self.y_test = (
            train_test_split(
                self.X,
                self.y,
                test_size=test_size,
                random_state=random_state,
                stratify=self.y,
            )
        )

        self.random_state = random_state

    def evaluate_features(self, selected_features=None):

        if selected_features is None:
            selected_features = self.X.columns.tolist()

        if len(selected_features) == 0:
            raise ValueError(
                "At least one feature must be selected."
            )

        X_train = self.X_train[selected_features]
        X_test = self.X_test[selected_features]

        # Build and train model.
        model = build_pipeline(X_train)

        model.fit(X_train, self.y_train)

        # Generate predictions.
        y_pred = model.predict(X_test)

        y_proba = model.predict_proba(X_test)[:, 1]

        # Calculate performance metrics.
        metrics = {
            "Accuracy": accuracy_score(
                self.y_test, y_pred
            ),

            "Precision": precision_score(
                self.y_test,
                y_pred,
                zero_division=0,
            ),

            "Recall": recall_score(
                self.y_test,
                y_pred,
                zero_division=0,
            ),

            "F1 Score": f1_score(
                self.y_test,
                y_pred,
                zero_division=0,
            ),

            "ROC-AUC": roc_auc_score(
                self.y_test,
                y_proba,
            ),

            "PR-AUC": average_precision_score(
                self.y_test,
                y_proba,
            ),

            "Number of Features": len(
                selected_features
            ),
        }

        return metrics, model

    def compare_features(self, selected_features):

        # Evaluate all available features.
        baseline_metrics, baseline_model = (
            self.evaluate_features()
        )

        # Evaluate RL-selected features.
        rl_metrics, rl_model = (
            self.evaluate_features(selected_features)
        )

        results = pd.DataFrame(
            [
                {
                    "Model": "All Features",
                    **baseline_metrics,
                },
                {
                    "Model": "RL Selected Features",
                    **rl_metrics,
                },
            ]
        )

        return results, baseline_model, rl_model

    def get_classification_report(
        self,
        model,
        selected_features,
    ):

        X_test = self.X_test[selected_features]

        predictions = model.predict(X_test)

        return classification_report(
            self.y_test,
            predictions,
            zero_division=0,
        )

    def cross_validation(
        self,
        selected_features=None,
        cv_splits=5,
    ):

        if selected_features is None:
            selected_features = self.X.columns.tolist()

        X_selected = self.X[selected_features]

        min_class_count = int(
            self.y.value_counts().min()
        )

        n_splits = min(
            cv_splits,
            min_class_count,
        )

        if n_splits < 2:
            raise ValueError(
                "Not enough examples in the minority class "
                "for cross-validation."
            )

        model = build_pipeline(X_selected)

        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        scores = cross_validate(
            model,
            X_selected,
            self.y,
            cv=cv,
            scoring=[
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "average_precision",
            ],
            n_jobs=-1,
        )

        return pd.DataFrame({
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROC-AUC",
                "PR-AUC",
            ],
            "Mean Score": [
                scores["test_accuracy"].mean(),
                scores["test_precision"].mean(),
                scores["test_recall"].mean(),
                scores["test_f1"].mean(),
                scores["test_roc_auc"].mean(),
                scores["test_average_precision"].mean(),
            ],
            "Std": [
                scores["test_accuracy"].std(),
                scores["test_precision"].std(),
                scores["test_recall"].std(),
                scores["test_f1"].std(),
                scores["test_roc_auc"].std(),
                scores["test_average_precision"].std(),
            ],
        })
