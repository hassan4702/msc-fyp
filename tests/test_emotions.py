from backend.emotions import EMOTIONS, Emotion


def test_seven_canonical_emotions():
    assert len(EMOTIONS) == 7
    assert EMOTIONS[-1] == "neutral"


def test_enum_values_match_list():
    assert [e.value for e in Emotion] == EMOTIONS
