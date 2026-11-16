"""The interface every FireMeNot model must follow.

Owner: Machine-Learning / Model Lead (interface agreed with QA)

Why: the evaluation code, the tests, and the app all call models the same way.
If your model follows this interface, it plugs into everything else for free.

Input  X: numpy array of preprocessed images, shape (n, H, W, 3), floats in [0, 1]
          (the output of src.preprocessing.pipeline.load_image, stacked)
Output:   predict_proba(X) -> shape (n,) array, probability of SMOKE (label 1)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SmokeClassifier(ABC):
    name: str = "unnamed-model"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SmokeClassifier":
        """Train on images X and labels y (1 = smoke, 0 = no smoke). Return self."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(smoke) for each image as a 1-D array of floats in [0, 1]."""

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


class MajorityBaseline(SmokeClassifier):
    """Always predicts the most common training class.

    This is the "dumb" floor. Any real model that can't beat it is broken.
    Also serves as a working example of the interface above.
    """

    name = "majority-baseline"

    def fit(self, X, y):
        self.p_smoke_ = float(np.mean(np.asarray(y) == 1))
        return self

    def predict_proba(self, X):
        return np.full(len(X), self.p_smoke_, dtype=float)
