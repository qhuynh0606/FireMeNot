## What does this PR do?
<!-- One or two sentences. Link the issue: "Closes #12" -->

## Type
- [ ] Preprocessing / dataset
- [ ] Model
- [ ] Integration / app
- [ ] Evaluation / tests
- [ ] Docs / other

## Checklist
- [ ] CI is green (lint + tests)
- [ ] I ran `pytest` locally
- [ ] New code has tests, or it makes a skipped contract test in `tests/test_contracts.py` pass
- [ ] No images, datasets, or model weight files are committed
- [ ] I didn't change a shared contract (image size, label convention, `predict_proba` output, split method) without telling the team

## If this changes a model: results
<!-- Required for model PRs. Use src/evaluation/metrics.py: paste format_report(...) output.
     Report VALIDATION results. Test-set numbers are only for the final report. -->

| Metric | Value |
|---|---|
| Accuracy | |
| Recall | |
| Specificity | |
| **False negative rate** | |

Threshold used: <!-- 0.5, or tuned on val with threshold_for_recall? -->

## Screenshots / notes for the reviewer
