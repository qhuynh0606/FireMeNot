import math

import numpy as np
import pytest

from src.evaluation.metrics import (
    confusion_counts,
    evaluate,
    format_report,
    log_results,
    threshold_for_recall,
)

# Hand-checked example: 4 smoke, 6 no-smoke
#   smoke:    caught 3, missed 1      -> TP=3, FN=1
#   no smoke: correct 5, false alarm 1 -> TN=5, FP=1
Y_TRUE = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
Y_PRED = [1, 1, 1, 0, 0, 0, 0, 0, 0, 1]


def test_confusion_counts_hand_example():
    assert confusion_counts(Y_TRUE, Y_PRED) == {"tp": 3, "tn": 5, "fp": 1, "fn": 1}


def test_evaluate_hand_example():
    r = evaluate(Y_TRUE, Y_PRED)
    assert r["n"] == 10
    assert r["accuracy"] == pytest.approx(0.8)
    assert r["recall"] == pytest.approx(0.75)
    assert r["specificity"] == pytest.approx(5 / 6)
    assert r["precision"] == pytest.approx(0.75)
    assert r["false_negative_rate"] == pytest.approx(0.25)
    assert r["false_positive_rate"] == pytest.approx(1 / 6)
    assert r["recall"] + r["false_negative_rate"] == pytest.approx(1.0)


def test_perfect_predictions():
    r = evaluate(Y_TRUE, Y_TRUE)
    assert r["accuracy"] == 1.0
    assert r["false_negative_rate"] == 0.0


def test_probabilities_are_thresholded():
    probs = [0.9, 0.6, 0.4, 0.1]
    r_default = evaluate([1, 1, 1, 0], y_prob=probs)
    assert (r_default["tp"], r_default["fn"]) == (2, 1)
    r_low = evaluate([1, 1, 1, 0], y_prob=probs, threshold=0.3)
    assert (r_low["tp"], r_low["fn"]) == (3, 0)


def test_roc_auc_when_probs_given():
    r = evaluate([0, 0, 1, 1], y_prob=[0.1, 0.2, 0.8, 0.9])
    assert r["roc_auc"] == pytest.approx(1.0)


def test_no_smoke_examples_gives_nan_recall_not_crash():
    r = evaluate([0, 0, 0], [0, 0, 1])
    assert math.isnan(r["recall"])
    assert r["specificity"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("y_true, y_pred", [
    ([], []),                  # empty
    ([1, 0], [1]),             # length mismatch
    ([1, 2], [1, 0]),          # invalid label
    (["smoke", "none"], [1, 0]),
])
def test_bad_inputs_raise(y_true, y_pred):
    with pytest.raises(ValueError):
        evaluate(y_true, y_pred)


def test_bad_probabilities_raise():
    with pytest.raises(ValueError):
        evaluate([1, 0], y_prob=[1.2, 0.1])
    with pytest.raises(ValueError):
        evaluate([1, 0])


def test_threshold_for_recall():
    y = [1, 1, 1, 1, 0, 0]
    p = [0.9, 0.8, 0.6, 0.3, 0.5, 0.1]
    t = threshold_for_recall(y, p, target_recall=0.75)
    assert t == pytest.approx(0.6)
    assert evaluate(y, y_prob=p, threshold=t)["recall"] >= 0.75
    assert threshold_for_recall(y, p, target_recall=1.0) == pytest.approx(0.3)


def test_log_results_appends_rows(tmp_path):
    path = tmp_path / "metrics.csv"
    r = evaluate(Y_TRUE, Y_PRED)
    log_results(r, "test-model", "val", path)
    log_results(r, "test-model", "test", path)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 3  # header + 2 rows
    assert "false_negative_rate" in lines[0]
    with pytest.raises(ValueError):
        log_results(r, "test-model", "training", path)


def test_format_report_contains_key_numbers():
    text = format_report(evaluate(Y_TRUE, Y_PRED), "demo")
    assert "False negative rate" in text
    assert "25.0%" in text
    assert "1 (FN)" in text


def test_numpy_inputs_work():
    r = evaluate(np.array(Y_TRUE), np.array(Y_PRED))
    assert r["tp"] == 3
