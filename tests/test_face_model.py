import importlib.util

import pytest

from backend.emotions import EMOTIONS
from backend.models.base import scores_from_logits
from backend.models.face_model import aggregate_frame_logits


def test_scores_from_logits_normalises_over_emotions():
    scores = scores_from_logits([0.0] * 7)
    assert set(scores) == set(EMOTIONS)
    assert abs(sum(scores.values()) - 1.0) < 1e-6


def test_aggregate_no_faces_is_unavailable():
    pred = aggregate_frame_logits([None, None])
    assert pred.available is False
    assert pred.source == "face"


def test_aggregate_single_detected_face():
    logits = [0.0] * 7
    logits[EMOTIONS.index("happy")] = 10.0
    pred = aggregate_frame_logits([logits, None])  # one frame had a face, one didn't
    assert pred.available is True
    assert pred.label == "happy"


def test_aggregate_averages_probabilities_over_frames():
    a = [0.0] * 7
    a[EMOTIONS.index("happy")] = 10.0
    b = [0.0] * 7
    b[EMOTIONS.index("sad")] = 10.0
    pred = aggregate_frame_logits([a, b])
    assert pred.probabilities["happy"] > 0.3
    assert pred.probabilities["sad"] > 0.3


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_facenet_forward_shape():
    import torch

    from backend.models.face_net import FaceNet

    out = FaceNet()(torch.zeros(2, 1, 48, 48))
    assert tuple(out.shape) == (2, 7)
