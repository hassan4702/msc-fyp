"""MELD emotion labels -> canonical 7 labels.

MELD already uses 7 emotions that line up almost exactly with ours; only "joy"
and "sadness" need renaming to "happy"/"sad". Mapping by name (case-insensitive)
since the CSVs capitalise inconsistently.
"""
from backend.emotions import EMOTIONS

MELD_EMOTIONS: list[str] = ["neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"]

MELD_TO_CANONICAL: dict[str, str] = {
    "neutral": "neutral",
    "joy": "happy",
    "sadness": "sad",
    "anger": "anger",
    "surprise": "surprise",
    "fear": "fear",
    "disgust": "disgust",
}

assert set(MELD_TO_CANONICAL) == set(MELD_EMOTIONS)
assert all(v in EMOTIONS for v in MELD_TO_CANONICAL.values())


def meld_to_canonical(emotion: str) -> str:
    return MELD_TO_CANONICAL[emotion.lower()]
