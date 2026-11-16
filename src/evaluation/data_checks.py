"""Data-quality checks for the FIgLib dataset.

Owner: QA & Evaluation Lead (written with the Dataset & Preprocessing Lead)

These catch the problems that silently ruin results:
  * bad or missing labels
  * duplicate images
  * leakage between train / val / test
  * preprocessed images with the wrong shape or value range
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np

# FIgLib frames are named like "1465065308_+00060.jpg": a Unix timestamp, then the
# offset in seconds from the fire's ignition. Negative offset = before ignition
# (no smoke), positive = after ignition (smoke).
# TODO(Dataset Lead): confirm this against the files you actually download, and
# decide with the team whether offset 0 counts as smoke.
FIGLIB_NAME = re.compile(r"^(?P<ts>\d+)_(?P<offset>[+-]\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def label_from_figlib_filename(name: str) -> int:
    """Return 1 (smoke) or 0 (no smoke) from a FIgLib frame filename."""
    m = FIGLIB_NAME.match(Path(name).name)
    if not m:
        raise ValueError(f"Not a FIgLib frame filename: {name!r}")
    return 1 if int(m.group("offset")) > 0 else 0


def check_labels(labels: Iterable, allowed: tuple = (0, 1)) -> list[int]:
    """Return the indices of labels that are missing or not in `allowed`."""
    bad = []
    for i, lab in enumerate(labels):
        if lab is None or (isinstance(lab, float) and np.isnan(lab)) or lab not in allowed:
            bad.append(i)
    return bad


def file_hash(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def find_duplicate_files(paths: Iterable[str | Path]) -> list[list[Path]]:
    """Group byte-identical files. Returns only groups with 2+ files."""
    by_hash: dict[str, list[Path]] = defaultdict(list)
    for p in paths:
        by_hash[file_hash(p)].append(Path(p))
    return [sorted(group) for group in by_hash.values() if len(group) > 1]


def check_split_leakage(split_groups: dict[str, Iterable[str]]) -> dict[tuple[str, str], set[str]]:
    """Find IDs that appear in more than one split.

    `split_groups` maps split name -> IDs in that split, e.g.
        {"train": [...], "val": [...], "test": [...]}

    Run it twice:
      1. with image IDs / file hashes  -> the same image is in two splits
      2. with FIRE EVENT IDs (the folder each sequence came from) -> frames from the
         same fire are in two splits. Neighboring frames are nearly identical, so
         splitting by frame instead of by fire inflates test scores.

    Returns {(split_a, split_b): overlapping_ids}; empty dict means no leakage.
    """
    sets = {name: set(ids) for name, ids in split_groups.items()}
    leaks = {}
    for a, b in combinations(sorted(sets), 2):
        overlap = sets[a] & sets[b]
        if overlap:
            leaks[(a, b)] = overlap
    return leaks


def check_image_array(arr: np.ndarray, expected_shape: tuple[int, ...],
                      value_range: tuple[float, float] = (0.0, 1.0)) -> list[str]:
    """Return a list of problems with a preprocessed image (empty list = OK)."""
    problems = []
    if arr.shape != tuple(expected_shape):
        problems.append(f"shape {arr.shape} != expected {tuple(expected_shape)}")
    if not np.issubdtype(arr.dtype, np.floating):
        problems.append(f"dtype {arr.dtype} is not floating point (did you forget to normalize?)")
    if arr.size and np.isnan(arr).any():
        problems.append("contains NaN values")
    lo, hi = value_range
    if arr.size and (arr.min() < lo or arr.max() > hi):
        problems.append(f"values outside [{lo}, {hi}]: min={arr.min():.3f}, max={arr.max():.3f}")
    return problems


def class_balance(labels: Iterable[int]) -> dict[int, float]:
    """Fraction of each label. Report this per split in the dataset docs."""
    arr = np.asarray(list(labels))
    if arr.size == 0:
        return {}
    values, counts = np.unique(arr, return_counts=True)
    return {int(v): float(c / arr.size) for v, c in zip(values, counts, strict=True)}
