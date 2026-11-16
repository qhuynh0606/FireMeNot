"""Shared evaluation metrics for FireMeNot.

Owner: QA & Evaluation Lead

EVERY model must be scored with `evaluate()` so results are comparable.
Do not write your own accuracy/recall code in a notebook and report that.

Label convention (used everywhere in this repo):
    1 = smoke
    0 = no smoke
"""

from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path

import numpy as np

SMOKE = 1
NO_SMOKE = 0


def _as_binary_array(values, name: str) -> np.ndarray:
    arr = np.asarray(values).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    bad = set(np.unique(arr).tolist()) - {0, 1}
    if bad:
        raise ValueError(f"{name} must contain only 0/1 labels, found {sorted(bad)}")
    return arr.astype(int)


def confusion_counts(y_true, y_pred) -> dict[str, int]:
    """Return true/false positive/negative counts (positive = smoke)."""
    y_true = _as_binary_array(y_true, "y_true")
    y_pred = _as_binary_array(y_pred, "y_pred")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    return {
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
    }


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def evaluate(y_true, y_pred=None, y_prob=None, threshold: float = 0.5) -> dict:
    """Score a model's predictions.

    Pass either hard predictions (`y_pred`), smoke probabilities (`y_prob`), or both.
    If only `y_prob` is given, predictions are `y_prob >= threshold`.

    Returns a dict with: n, tp, tn, fp, fn, accuracy, recall, specificity,
    precision, f1, false_negative_rate, false_positive_rate, threshold, and
    roc_auc (only when y_prob is given and both classes are present).
    """
    if y_pred is None and y_prob is None:
        raise ValueError("Provide y_pred, y_prob, or both")

    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float).ravel()
        if np.any((y_prob < 0) | (y_prob > 1)) or np.any(np.isnan(y_prob)):
            raise ValueError("y_prob must be probabilities in [0, 1] with no NaNs")
        if y_pred is None:
            y_pred = (y_prob >= threshold).astype(int)

    c = confusion_counts(y_true, y_pred)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    n = tp + tn + fp + fn

    recall = _safe_div(tp, tp + fn)          # a.k.a. sensitivity / true positive rate
    precision = _safe_div(tp, tp + fp)
    f1 = (
        float("nan")
        if math.isnan(recall) or math.isnan(precision) or (recall + precision) == 0
        else 2 * precision * recall / (precision + recall)
    )

    results = {
        "n": n,
        **c,
        "accuracy": _safe_div(tp + tn, n),
        "recall": recall,
        "specificity": _safe_div(tn, tn + fp),
        "precision": precision,
        "f1": f1,
        # Missed smoke = delayed wildfire detection. This is the number we care most about.
        "false_negative_rate": _safe_div(fn, tp + fn),
        "false_positive_rate": _safe_div(fp, fp + tn),
        "threshold": threshold if y_prob is not None else None,
        "roc_auc": float("nan"),
    }

    if y_prob is not None:
        y_true_arr = _as_binary_array(y_true, "y_true")
        if len(np.unique(y_true_arr)) == 2:
            from sklearn.metrics import roc_auc_score

            results["roc_auc"] = float(roc_auc_score(y_true_arr, y_prob))

    return results


def threshold_for_recall(y_true, y_prob, target_recall: float = 0.95) -> float:
    """Highest decision threshold that still reaches `target_recall` on smoke images.

    Use on the VALIDATION set (never the test set) to tune how cautious a model is.
    Lower thresholds catch more smoke (fewer false negatives) at the cost of more
    false alarms.
    """
    y_true = _as_binary_array(y_true, "y_true")
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1]")
    pos_probs = y_prob[y_true == SMOKE]
    if pos_probs.size == 0:
        raise ValueError("Need at least one smoke example to tune a recall threshold")

    for t in np.sort(np.unique(y_prob))[::-1]:
        if np.mean(pos_probs >= t) >= target_recall:
            return float(t)
    return float(np.min(y_prob))


RESULT_FIELDS = [
    "timestamp", "model", "split", "n", "accuracy", "recall", "specificity",
    "precision", "f1", "false_negative_rate", "false_positive_rate", "roc_auc",
    "tp", "tn", "fp", "fn", "threshold", "notes",
]


def log_results(results: dict, model_name: str, split: str,
                path: str | Path = "results/metrics.csv", notes: str = "") -> Path:
    """Append one row to the shared results table so every experiment is recorded."""
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be 'train', 'val', or 'test'")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()

    row = {k: results.get(k, "") for k in RESULT_FIELDS}
    row.update(timestamp=datetime.now().isoformat(timespec="seconds"),
               model=model_name, split=split, notes=notes)

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    return path


def format_report(results: dict, model_name: str) -> str:
    """Human-readable summary to paste into a PR description."""
    def pct(x):
        return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.1%}"

    auc = results["roc_auc"]
    auc_str = "n/a" if math.isnan(auc) else f"{auc:.3f}"

    return "\n".join([
        f"### {model_name}",
        "| Metric | Value |",
        "|---|---|",
        f"| Accuracy | {pct(results['accuracy'])} |",
        f"| Recall (smoke caught) | {pct(results['recall'])} |",
        f"| Specificity | {pct(results['specificity'])} |",
        f"| Precision | {pct(results['precision'])} |",
        f"| **False negative rate** | **{pct(results['false_negative_rate'])}** |",
        f"| ROC AUC | {auc_str} |",
        "",
        "Confusion matrix (rows = actual, cols = predicted):",
        "",
        "| | Pred no smoke | Pred smoke |",
        "|---|---|---|",
        f"| **Actual no smoke** | {results['tn']} (TN) | {results['fp']} (FP) |",
        f"| **Actual smoke** | {results['fn']} (FN) | {results['tp']} (TP) |",
    ])
