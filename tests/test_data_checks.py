import numpy as np
import pytest

from src.evaluation.data_checks import (
    check_image_array,
    check_labels,
    check_split_leakage,
    class_balance,
    find_duplicate_files,
    label_from_figlib_filename,
)


@pytest.mark.parametrize("name, label", [
    ("1465065308_+00060.jpg", 1),
    ("1465065308_-00120.jpg", 0),
    ("1465065308_+00000.jpg", 0),
    ("some/folder/1465065308_+02400.JPG", 1),
])
def test_figlib_labels(name, label):
    assert label_from_figlib_filename(name) == label


def test_figlib_bad_name():
    with pytest.raises(ValueError):
        label_from_figlib_filename("IMG_1234.jpg")


def test_check_labels():
    assert check_labels([0, 1, 1, 0]) == []
    assert check_labels([0, 2, None, 1, float("nan")]) == [1, 2, 4]


def test_find_duplicates(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"same")
    (tmp_path / "b.jpg").write_bytes(b"same")
    (tmp_path / "c.jpg").write_bytes(b"different")
    dups = find_duplicate_files(tmp_path.glob("*.jpg"))
    assert len(dups) == 1
    assert [p.name for p in dups[0]] == ["a.jpg", "b.jpg"]


def test_split_leakage_detected():
    leaks = check_split_leakage({
        "train": ["fire1", "fire2", "fire3"],
        "val": ["fire4"],
        "test": ["fire3", "fire5"],
    })
    assert leaks == {("test", "train"): {"fire3"}}


def test_no_leakage():
    assert check_split_leakage({"train": ["a"], "val": ["b"], "test": ["c"]}) == {}


def test_image_array_ok():
    img = np.random.default_rng(0).random((224, 224, 3), dtype=np.float32)
    assert check_image_array(img, (224, 224, 3)) == []


def test_image_array_problems():
    raw = np.full((100, 100, 3), 255, dtype=np.uint8)  # not resized, not normalized
    problems = check_image_array(raw, (224, 224, 3))
    assert len(problems) == 3  # shape, dtype, range


def test_class_balance():
    assert class_balance([1, 0, 0, 0]) == {0: 0.75, 1: 0.25}
    assert class_balance([]) == {}
