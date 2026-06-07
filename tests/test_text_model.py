from backend.emotions import EMOTIONS
from backend.models.text_model import _scores_from_logits


def test_scores_from_logits_normalises_over_emotions():
    scores = _scores_from_logits([0.0] * 7)
    assert set(scores) == set(EMOTIONS)
    assert abs(sum(scores.values()) - 1.0) < 1e-6


def test_scores_from_logits_argmax_matches_largest_logit():
    logits = [0.0] * 7
    logits[EMOTIONS.index("sad")] = 10.0
    scores = _scores_from_logits(logits)
    assert max(scores, key=scores.get) == "sad"
