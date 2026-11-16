# FireMeNot™: Wildfire Smoke Detection System

![tests](../../actions/workflows/tests.yml/badge.svg)

Machine-learning system that analyzes wildfire camera images and predicts whether smoke
is present, along with a confidence score. Built for SDSU CS450.

## Motivation
Wildfires are a recurring hazard in California. Earlier smoke detection means earlier
response. We compare classic ML methods and a CNN to see which detect smoke most
reliably, with special focus on **missed smoke (false negatives)**.

## Dataset
[HPWREN Fire Ignition Library (FIgLib)](https://www.hpwren.ucsd.edu/FIgLib/): about 24,000
images from wildfire cameras across Southern California, covering both smoke and no-smoke
conditions.

> The dataset is **not** stored in this repo. Download it into `data/` (see
> `src/preprocessing/`). *Download script coming soon.*

## Models
| Model | Status |
|---|---|
| Majority-class baseline | ✅ |
| Logistic regression | 🚧 |
| Support vector machine | 🚧 |
| CNN / transfer learning | 🚧 |

## Results
*Validation results will be filled in from `results/metrics.csv`.*

| Model | Accuracy | Recall | Specificity | False negative rate |
|---|---|---|---|---|
| | | | | |

See [`docs/evaluation_plan.md`](docs/evaluation_plan.md) for how models are evaluated.

## Project structure
```
src/
  preprocessing/   load, resize, normalize, augment, split
  models/          SmokeClassifier interface + each model
  evaluation/      shared metrics + data-quality checks
  app/             single-image prediction / alert
tests/             unit + contract tests (run with pytest)
docs/              evaluation plan and project docs
results/           shared metrics table
```

## Getting started
```bash
pip install -r requirements.txt
pytest -rs     # skipped tests = work still to be done
```
See [CONTRIBUTING.md](CONTRIBUTING.md) for workflow and team rules.

## Team
| Name | Role |
|---|---|
| | Project Manager / Repository Coordinator |
| | Dataset & Preprocessing Lead |
| Jania Little | Machine-Learning / Model Lead |
| Fabian Llamas | Software Integration Lead |
| Quoc-Jonathan Huynh | QA & Evaluation Lead |
