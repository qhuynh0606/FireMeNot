"""Support Vector Machine comparison model.

Owner: Machine-Learning / Model Lead
Status: TODO

Hints:
  * Same flattening / feature step as the logistic-regression baseline. Reuse it.
  * sklearn.svm.SVC(kernel="rbf", probability=True) gives predict_proba.
    probability=True is slower; for big data consider LinearSVC + CalibratedClassifierCV.
  * SVMs are slow on ~24k full-size images. Downsample images or use extracted features.
"""

from __future__ import annotations

from .base import SmokeClassifier


class SVMModel(SmokeClassifier):
    name = "svm"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y):
        raise NotImplementedError("TODO(ML Lead): train the SVM")

    def predict_proba(self, X):
        raise NotImplementedError("TODO(ML Lead): return P(smoke) as a 1-D array")
