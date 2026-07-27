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


@pytest.mark.skipif(
    importlib.util.find_spec("mediapipe") is None or importlib.util.find_spec("cv2") is None,
    reason="mediapipe/cv2 not installed",
)
@pytest.mark.parametrize("box", [(200, 150, 120, 120), (4, 4, 60, 60)])  # centred, and at the edge
def test_detector_crop_applies_margin_and_clamps_at_the_border(monkeypatch, box):
    """The margin must widen the detected box, and must never run off the image edge.

    Training and inference share this function, so a change here silently re-framing the
    face is exactly the bug class that cost 6pp of accuracy before. Pin the geometry.
    """
    import numpy as np

    from backend.models.face_detect import MARGIN, FaceDetector

    detector = FaceDetector()
    x, y, w, h = box
    fake = type("B", (), {"origin_x": x, "origin_y": y, "width": w, "height": h})()
    monkeypatch.setattr(
        detector, "_detector",
        type("D", (), {"detect": lambda *a: type("R", (), {"detections": [type("F", (), {"bounding_box": fake})()]})()})(),
    )

    img = np.random.default_rng(0).integers(0, 255, (480, 640), dtype=np.uint8)
    crop = detector.crop(img)

    mx, my = int(w * MARGIN), int(h * MARGIN)
    expected = (min(480, y + h + my) - max(0, y - my), min(640, x + w + mx) - max(0, x - mx))
    assert crop.shape == expected
    assert crop.shape[0] > h and crop.shape[1] > w  # the margin actually widened it
    assert crop.size > 0  # never empty, even clamped at the border


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_facenet_forward_shape():
    import torch

    from backend.models.face_net import FaceNet

    out = FaceNet()(torch.zeros(2, 1, 48, 48))
    assert tuple(out.shape) == (2, 7)
