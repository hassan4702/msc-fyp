import importlib.util
import json

import pytest

from backend.emotions import EMOTIONS
from backend.models.base import read_temperature, scores_from_logits
from training.calibrate import expected_calibration_error


def test_temperature_softens_distribution():
    logits = [0.0] * 7
    logits[EMOTIONS.index("happy")] = 5.0
    sharp = scores_from_logits(logits, temperature=1.0)
    soft = scores_from_logits(logits, temperature=4.0)
    assert soft["happy"] < sharp["happy"]                          # higher T -> less peaked
    assert max(soft, key=soft.get) == max(sharp, key=sharp.get)    # argmax unchanged
    assert abs(sum(soft.values()) - 1.0) < 1e-6


def test_temperature_one_is_identity():
    logits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    assert scores_from_logits(logits) == scores_from_logits(logits, temperature=1.0)


def test_read_temperature_default_when_absent(tmp_path):
    assert read_temperature(str(tmp_path)) == 1.0


def test_read_temperature_from_file(tmp_path):
    (tmp_path / "calibration.json").write_text(json.dumps({"temperature": 2.5}))
    assert read_temperature(str(tmp_path)) == 2.5


def test_ece_perfectly_calibrated_is_zero():
    assert expected_calibration_error([1.0, 1.0, 1.0, 1.0], [True, True, True, True]) < 1e-9


def test_ece_overconfident_is_high():
    assert expected_calibration_error([0.9, 0.9, 0.9, 0.9], [False, False, False, False]) > 0.8


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_fit_temperature_softens_overconfident_logits():
    from training.calibrate import fit_temperature

    logits, labels = [], []
    for i in range(140):
        vec = [0.0] * 7
        true = i % 7
        pred = true if i % 10 >= 4 else (true + 1) % 7  # 40% wrong but very confident
        vec[pred] = 8.0
        logits.append(vec)
        labels.append(true)
    assert fit_temperature(logits, labels) > 1.0
