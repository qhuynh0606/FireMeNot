"""Model registry. Add new models here so the contract tests pick them up."""

from .base import MajorityBaseline, SmokeClassifier
from .cnn import CNNModel
from .logistic_regression import LogisticRegressionModel
from .svm import SVMModel

ALL_MODELS: list[type[SmokeClassifier]] = [
    MajorityBaseline,
    LogisticRegressionModel,
    SVMModel,
    CNNModel,
]

__all__ = ["ALL_MODELS", "SmokeClassifier", "MajorityBaseline",
           "LogisticRegressionModel", "SVMModel", "CNNModel"]
