"""Image loading, preprocessing, and dataset splits.

Owner: Dataset & Preprocessing Lead
Status: TODO. tests/test_contracts.py checks these once implemented.

Agreed contract (change it only by telling the whole team):
  * IMAGE_SIZE = (224, 224). Matches common pretrained CNNs.
  * load_image() returns float32 array, shape (224, 224, 3), values in [0, 1], RGB.
  * Splits are made BY FIRE EVENT, not by individual frame, so near-identical
    frames from the same fire never land in both train and test.
  * Splits must be reproducible: same seed -> same split.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

IMAGE_SIZE = (224, 224)
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 450


def load_image(path: str | Path, size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Load one image -> float32 RGB array of shape (size[1], size[0], 3) in [0, 1].

    Hints: PIL.Image.open(path).convert("RGB").resize(size), then np.asarray(...) / 255.0
    """
    raise NotImplementedError("TODO(Dataset Lead): load, resize, and normalize an image")


def make_splits(event_ids: list[str], ratios: dict[str, float] = SPLIT_RATIOS,
                seed: int = SEED) -> dict[str, list[str]]:
    """Assign each FIRE EVENT (not each image) to train / val / test.

    Returns {"train": [event ids], "val": [...], "test": [...]}; every event
    appears in exactly one split.

    Hints: np.random.default_rng(seed).permutation(sorted(set(event_ids))), then slice.
    """
    raise NotImplementedError("TODO(Dataset Lead): split fire events into train/val/test")


def augment(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random augmentation for TRAINING images only (flip, crop, brightness...).

    Must return the same shape and keep values in [0, 1]. Never augment val/test.
    """
    raise NotImplementedError("TODO(Dataset Lead): add augmentation (optional, after the basics work)")
