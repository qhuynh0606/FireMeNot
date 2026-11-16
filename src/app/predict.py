"""Turn a trained model into a usable prediction / alert.

Owner: Software Integration Lead
Status: TODO. tests/test_contracts.py checks this once implemented.

Agreed contract:
  predict_image(path, model, threshold) -> {
      "smoke": bool,          # True if P(smoke) >= threshold
      "confidence": float,    # P(smoke), in [0, 1]
      "threshold": float,
      "image": str,           # the input path
  }
"""

from __future__ import annotations

from pathlib import Path

from src.models.base import SmokeClassifier


def predict_image(path: str | Path, model: SmokeClassifier, threshold: float = 0.5) -> dict:
    """Load + preprocess one image with src.preprocessing.pipeline.load_image,
    run model.predict_proba on a batch of one, and return the dict above.
    """
    raise NotImplementedError("TODO(Integration Lead): wire preprocessing + model into one call")
