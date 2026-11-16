# FireMeNot Evaluation Plan

**Owner:** QA & Evaluation Lead
**Status:** Draft. Team: comment or open a PR to change anything here.

## 1. Goal
Measure how accurately each model separates **smoke** from **no-smoke** images, with
particular attention to **missed smoke (false negatives)**. A missed detection means a
delayed wildfire response; a false alarm only costs someone a second look.

## 2. Label convention
`1 = smoke`, `0 = no smoke`, used in every file, model, and table.

## 3. Data splits
| Split | Share | Used for |
|---|---|---|
| Train | 70% | Fitting models |
| Validation | 15% | Comparing models, tuning hyperparameters and thresholds |
| Test | 15% | **Final report only.** Evaluated once per final model |

- Splits are made **by fire event**, not by frame. FIgLib frames from the same fire are
  nearly identical, so frame-level splitting leaks test data into training and inflates scores.
- Fixed seed (`SEED = 450`) so everyone gets the same split.
- Checked automatically by `check_split_leakage()` and `test_splits_contract`.

## 4. Metrics
All computed by `src/evaluation/metrics.py::evaluate()`. No hand-rolled metrics in reports.

| Metric | Formula | Why |
|---|---|---|
| Accuracy | (TP+TN) / all | Overall correctness |
| **Recall** | TP / (TP+FN) | Share of smoke images caught |
| Specificity | TN / (TN+FP) | Share of clear images correctly left alone |
| Precision | TP / (TP+FP) | How trustworthy an alert is |
| **False negative rate** | FN / (TP+FN) | Primary risk metric |
| ROC AUC | | Threshold-independent ranking quality |
| Confusion matrix | | Required in every model report |

## 5. Models compared
1. Majority-class baseline (floor; every model must beat it)
2. Logistic regression (baseline)
3. SVM
4. CNN / transfer learning

## 6. Thresholds
- Default threshold: 0.5.
- Each model may also report a tuned threshold from `threshold_for_recall(target_recall=0.95)`,
  tuned on **validation only**.

## 7. Test cases (from the proposal's intended experiments)
<!-- TODO(QA): fill in once we've looked at the data -->
| ID | Condition | How we check it |
|---|---|---|
| TC-01 | Early, faint smoke (small positive offsets) | Recall on frames within the first N minutes after ignition |
| TC-02 | Clouds / fog / haze (likely false positives) | False positive rate on hand-picked no-smoke images |
| TC-03 | Different camera sites | Metrics broken down by camera |
| TC-04 | Time of day / lighting | Metrics broken down by time of day |
| TC-05 | Augmentation on vs off | Same model, compare validation metrics |

## 8. Acceptance criteria
<!-- TODO(team): agree on numbers -->
- Every model beats the majority baseline on validation accuracy.
- Final model false negative rate ≤ ___% on the test set.
- No split leakage (automated check passes).
- All results logged in `results/metrics.csv`.

## 9. Review process
- Every PR to `main` needs green CI and one approving review.
- Model PRs must include validation results in the PR template table.
- Defects are filed with the **Defect** issue template and tracked on the project board.
