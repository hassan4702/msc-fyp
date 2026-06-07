from backend.emotions import EMOTIONS
from backend.models.base import EmotionPrediction
from backend.models.fusion import ConfidenceGatedFusion, WeightedAverageFusion


def _pred(label: str, conf: float, source: str, available: bool = True) -> EmotionPrediction:
    """Build a prediction peaked on `label` with the given confidence."""
    rest = (1 - conf) / (len(EMOTIONS) - 1)
    scores = {e: rest for e in EMOTIONS}
    scores[label] = conf
    return EmotionPrediction.from_scores(scores, source, available)


def test_weighted_average_blends_both():
    text = _pred("happy", 0.8, "text")
    face = _pred("sad", 0.8, "face")
    res = WeightedAverageFusion(text_weight=0.5).fuse(text, face)
    assert res.prediction.source == "fused"
    assert res.prediction.probabilities["happy"] > 0
    assert res.prediction.probabilities["sad"] > 0


def test_falls_back_to_text_when_face_unavailable():
    text = _pred("anger", 0.9, "text")
    face = EmotionPrediction.unavailable("face")
    res = ConfidenceGatedFusion().fuse(text, face)
    assert res.prediction.label == "anger"
    assert res.face_weight == 0.0


def test_conflict_flagged_on_confident_disagreement():
    text = _pred("happy", 0.9, "text")
    face = _pred("anger", 0.9, "face")
    res = ConfidenceGatedFusion(conflict_threshold=0.5).fuse(text, face)
    assert res.conflicted is True


def test_no_conflict_when_modalities_agree():
    text = _pred("sad", 0.9, "text")
    face = _pred("sad", 0.9, "face")
    res = ConfidenceGatedFusion().fuse(text, face)
    assert res.conflicted is False


def test_confidence_gating_favours_more_confident_modality():
    text = _pred("happy", 0.9, "text")
    face = _pred("sad", 0.5, "face")
    res = ConfidenceGatedFusion().fuse(text, face)
    # Higher-confidence text should dominate the fused label.
    assert res.prediction.label == "happy"
    assert res.text_weight > res.face_weight
