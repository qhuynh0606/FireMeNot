"""Logistic-regression baseline.

Owner: Machine-Learning / Model Lead
Status: TODO. The contract tests in tests/test_contracts.py start running
automatically once fit/predict_proba stop raising NotImplementedError.

Hints:
  * Images are (n, H, W, 3). Logistic regression needs 2-D input, so flatten:
        X.reshape(len(X), -1)
    or extract simpler features first (e.g. color histograms, downsampled grayscale).
  * sklearn.linear_model.LogisticRegression(max_iter=1000); try class_weight="balanced".
  * Wrap it in a sklearn Pipeline with StandardScaler.
  * predict_proba must return ONLY the smoke column: model.predict_proba(X)[:, 1]
"""

from __future__ import annotations

from .base import SmokeClassifier


class LogisticRegressionModel(SmokeClassifier):
    name = "logistic-regression"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y):
        raise NotImplementedError("TODO(ML Lead): train the logistic-regression baseline")

    def predict_proba(self, X):
        raise NotImplementedError("TODO(ML Lead): return P(smoke) as a 1-D array")
