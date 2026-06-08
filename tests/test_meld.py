from backend.emotions import EMOTIONS
from backend.models.base import EmotionPrediction
from backend.models.fusion import ConfidenceGatedFusion
from evaluation.meld import MELD_EMOTIONS, MELD_TO_CANONICAL, meld_to_canonical
from evaluation.scoring import compare_systems, macro_f1


def _pred(label: str, conf: float, source: str, available: bool = True) -> EmotionPrediction:
    rest = (1 - conf) / (len(EMOTIONS) - 1)
    scores = {e: rest for e in EMOTIONS}
    scores[label] = conf
    return EmotionPrediction.from_scores(scores, source, available)


def test_meld_labels_map_to_canonical():
    assert set(MELD_TO_CANONICAL) == set(MELD_EMOTIONS)
    assert all(v in EMOTIONS for v in MELD_TO_CANONICAL.values())
    assert {MELD_TO_CANONICAL[m] for m in MELD_EMOTIONS} == set(EMOTIONS)  # covers all 7


def test_meld_key_mappings_case_insensitive():
    assert meld_to_canonical("joy") == "happy"
    assert meld_to_canonical("sadness") == "sad"
    assert meld_to_canonical("Neutral") == "neutral"


def test_macro_f1_perfect_on_present_labels():
    gold = ["happy", "sad", "anger", "happy"]
    assert macro_f1(gold, gold, sorted(set(gold))) == 1.0


def test_macro_f1_penalises_errors():
    gold = ["happy", "sad"]
    preds = ["happy", "happy"]
    assert 0.0 < macro_f1(gold, preds, sorted(set(gold))) < 1.0


def test_compare_systems_text_perfect_beats_face_and_majority():
    # one record per emotion; text always right, face always "neutral"
    records = [
        {"gold": e, "text": _pred(e, 0.9, "text"), "face": _pred("neutral", 0.9, "face")}
        for e in EMOTIONS
    ]
    out = compare_systems(records, ConfidenceGatedFusion())
    assert out["text_only"] == 1.0
    assert out["face_only"] < out["text_only"]
    assert out["majority"] < out["text_only"]
