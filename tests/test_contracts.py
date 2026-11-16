"""Contract tests: what each part of the system MUST do.

These are written ahead of the code. While a function still raises
NotImplementedError its tests are SKIPPED (shown as "s" in pytest output).
As soon as it's implemented, the tests run for real. If they fail, the
implementation doesn't match what the rest of the team is relying on.

Run just these:  pytest tests/test_contracts.py -v -rs
"""

import numpy as np
import pytest

from src.evaluation.data_checks import check_image_array, check_split_leakage
from src.evaluation.metrics import evaluate
from src.models import ALL_MODELS
from src.preprocessing import pipeline


def call_or_skip(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as e:
        pytest.skip(f"not implemented yet: {e}")


# ---------- tiny synthetic dataset (no real data needed in CI) ----------

@pytest.fixture
def tiny_dataset():
    """40 small images: 'smoke' images are brighter/grayer on top."""
    rng = np.random.default_rng(0)
    n = 40
    y = np.array([1] * 20 + [0] * 20)
    X = rng.random((n, 32, 32, 3)).astype(np.float32) * 0.5
    X[y == 1, :16] += 0.4
    return np.clip(X, 0, 1), y


@pytest.fixture
def sample_image(tmp_path):
    from PIL import Image

    arr = (np.random.default_rng(1).random((300, 400, 3)) * 255).astype(np.uint8)
    path = tmp_path / "1465065308_+00060.jpg"
    Image.fromarray(arr).save(path)
    return path


# ---------- Dataset & Preprocessing Lead ----------

def test_load_image_contract(sample_image):
    img = call_or_skip(pipeline.load_image, sample_image)
    h, w = pipeline.IMAGE_SIZE[1], pipeline.IMAGE_SIZE[0]
    assert check_image_array(img, (h, w, 3)) == []
    assert img.dtype == np.float32


def test_splits_contract():
    events = [f"fire{i}" for i in range(40)]
    splits = call_or_skip(pipeline.make_splits, events)
    assert set(splits) == {"train", "val", "test"}
    assert check_split_leakage(splits) == {}, "a fire event is in more than one split"
    assert sorted(sum(splits.values(), [])) == sorted(events), "events lost or duplicated"
    assert len(splits["train"]) > len(splits["val"])
    assert pipeline.make_splits(events) == splits, "splits are not reproducible"


def test_augment_contract():
    img = np.random.default_rng(0).random((224, 224, 3)).astype(np.float32)
    out = call_or_skip(pipeline.augment, img, np.random.default_rng(0))
    assert check_image_array(out, img.shape) == []


# ---------- Machine-Learning / Model Lead ----------

@pytest.mark.parametrize("model_cls", ALL_MODELS, ids=lambda c: c.name)
def test_model_contract(model_cls, tiny_dataset):
    X, y = tiny_dataset
    model = model_cls()
    fitted = call_or_skip(model.fit, X, y)
    assert fitted is model, "fit() must return self"

    probs = call_or_skip(model.predict_proba, X)
    assert probs.shape == (len(X),), "predict_proba must return a 1-D array (smoke column only)"
    assert np.all((probs >= 0) & (probs <= 1))

    preds = model.predict(X)
    assert set(np.unique(preds)) <= {0, 1}
    evaluate(y, preds, probs)  # must be scoreable by the shared evaluator


@pytest.mark.parametrize("model_cls", [m for m in ALL_MODELS if m.name != "majority-baseline"],
                         ids=lambda c: c.name)
def test_model_beats_majority_baseline(model_cls, tiny_dataset):
    """On an easy synthetic task, any real model should beat 'always guess one class'."""
    X, y = tiny_dataset
    model = model_cls()
    call_or_skip(model.fit, X, y)
    probs = call_or_skip(model.predict_proba, X)
    assert evaluate(y, y_prob=probs)["accuracy"] > 0.5


# ---------- Software Integration Lead ----------

def test_predict_image_contract(sample_image):
    from src.app.predict import predict_image
    from src.models.base import MajorityBaseline

    model = MajorityBaseline().fit(np.zeros((4, 1)), np.array([1, 1, 1, 0]))
    out = call_or_skip(predict_image, sample_image, model, 0.5)
    assert set(out) >= {"smoke", "confidence", "threshold", "image"}
    assert isinstance(out["smoke"], bool)
    assert 0.0 <= out["confidence"] <= 1.0
    assert out["smoke"] == (out["confidence"] >= out["threshold"])
