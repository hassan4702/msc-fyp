from backend.emotions import EMOTIONS
from training.fer2013 import FER_LABELS, FER_TO_CANONICAL, canonical_index_for_fer_name


def test_seven_fer_labels():
    assert len(FER_LABELS) == 7


def test_every_fer_label_maps_to_a_canonical_emotion():
    assert set(FER_TO_CANONICAL) == set(FER_LABELS)
    for mapped in FER_TO_CANONICAL.values():
        assert mapped in EMOTIONS


def test_mapping_covers_all_canonical_labels():
    assert {FER_TO_CANONICAL[label] for label in FER_LABELS} == set(EMOTIONS)


def test_canonical_index_for_fer_name():
    assert canonical_index_for_fer_name("angry") == EMOTIONS.index("anger")
    assert canonical_index_for_fer_name("happy") == EMOTIONS.index("happy")
    assert canonical_index_for_fer_name("neutral") == EMOTIONS.index("neutral")
