"""CNN / transfer-learning model.

Owner: Machine-Learning / Model Lead
Status: TODO

Hints:
  * Start from a pretrained backbone (e.g. torchvision ResNet18 or MobileNetV3),
    replace the final layer with a single output + sigmoid, and fine-tune.
  * Pretrained ImageNet models expect ImageNet normalization; convert from the
    [0, 1] images our pipeline produces inside this class, not in preprocessing.
  * Heavy deps (torch) are NOT in requirements.txt so CI stays fast. Import them
    inside methods and add them to requirements-cnn.txt.
  * Save trained weights to models/ (gitignored). Never commit weight files.
"""

from __future__ import annotations

from .base import SmokeClassifier


class CNNModel(SmokeClassifier):
    name = "cnn-transfer"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y):
        raise NotImplementedError("TODO(ML Lead): fine-tune a pretrained CNN")

    def predict_proba(self, X):
        raise NotImplementedError("TODO(ML Lead): return P(smoke) as a 1-D array")
