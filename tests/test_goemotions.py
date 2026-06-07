from backend.emotions import EMOTIONS
from training.goemotions import GOEMOTION_TO_EKMAN, GOEMOTIONS_LABELS, single_ekman_label


def test_all_28_goemotions_labels_present():
    assert len(GOEMOTIONS_LABELS) == 28
    assert GOEMOTIONS_LABELS[27] == "neutral"


def test_every_goemotion_maps_to_a_canonical_emotion():
    assert set(GOEMOTION_TO_EKMAN) == set(GOEMOTIONS_LABELS)
    for mapped in GOEMOTION_TO_EKMAN.values():
        assert mapped in EMOTIONS


def test_key_mappings_follow_official_ekman_grouping():
    assert GOEMOTION_TO_EKMAN["joy"] == "happy"
    assert GOEMOTION_TO_EKMAN["sadness"] == "sad"
    assert GOEMOTION_TO_EKMAN["annoyance"] == "anger"
    assert GOEMOTION_TO_EKMAN["disapproval"] == "anger"
    assert GOEMOTION_TO_EKMAN["nervousness"] == "fear"
    assert GOEMOTION_TO_EKMAN["curiosity"] == "surprise"
    assert GOEMOTION_TO_EKMAN["gratitude"] == "happy"
    assert GOEMOTION_TO_EKMAN["disgust"] == "disgust"
    assert GOEMOTION_TO_EKMAN["neutral"] == "neutral"


def test_single_label_for_one_emotion():
    joy = GOEMOTIONS_LABELS.index("joy")
    assert single_ekman_label([joy]) == "happy"


def test_single_label_keeps_same_bucket_multilabel():
    anger = GOEMOTIONS_LABELS.index("anger")
    annoyance = GOEMOTIONS_LABELS.index("annoyance")  # both -> anger
    assert single_ekman_label([anger, annoyance]) == "anger"


def test_single_label_drops_cross_bucket_multilabel():
    joy = GOEMOTIONS_LABELS.index("joy")      # happy
    anger = GOEMOTIONS_LABELS.index("anger")  # anger
    assert single_ekman_label([joy, anger]) is None


def test_single_label_none_for_empty():
    assert single_ekman_label([]) is None
