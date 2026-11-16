# Contributing to FireMeNot

## Where to start (your TODOs are already waiting)
Run:
```bash
pytest -rs
```
Every **SKIPPED** line is an unfinished piece of the project, labeled with who owns it.
Search the code for `TODO(` to find the exact functions. Each stub's docstring explains
what it must do. Your work is done when your skipped tests turn into passing tests.

| Role | Your files | Your tests |
|---|---|---|
| Dataset & Preprocessing | `src/preprocessing/pipeline.py` | `test_load_image_contract`, `test_splits_contract`, `test_augment_contract` |
| ML / Model | `src/models/*.py` | `test_model_contract`, `test_model_beats_majority_baseline` |
| Software Integration | `src/app/predict.py` | `test_predict_image_contract` |
| QA & Evaluation | `src/evaluation/`, `tests/`, `docs/evaluation_plan.md` | everything else |

## Setup
```bash
git clone <repo-url>
cd FireMeNot
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # add requirements-cnn.txt if you're working on the CNN
pytest
```

## Workflow
1. Pick or get assigned an issue.
2. Branch off `main`: `git checkout -b <area>/<short-description>`
   (e.g. `preprocessing/load-image`, `model/logreg-baseline`, `fix/split-leak`)
3. Commit small and often. Run `ruff check .` and `pytest` before pushing.
4. Open a pull request and fill in the template.
5. CI must be green, and you need one approval. Then merge.

**Never push directly to `main`.**

## Rules that keep us from breaking each other
- Labels: `1 = smoke`, `0 = no smoke`.
- Images: `float32`, shape `(224, 224, 3)`, values `[0, 1]`.
- Models: subclass `SmokeClassifier`; `predict_proba` returns a 1-D array of P(smoke).
- Splits are by fire event.
- Score with `src/evaluation/metrics.py`; log with `log_results(...)`.
- Don't commit data, images, or model weights. CI will fail if you do.
- Changing one of these? Post in the team chat first.
