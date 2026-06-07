"""FER-2013 (7 labels) -> canonical 7 Ekman labels.

FER-2013's native label order matches our canonical order; the only difference is
the name "angry" vs our "anger". We still map explicitly (not by position) so the
training head is guaranteed to output in EMOTIONS order regardless of how a given
dataset mirror happens to order its classes.
"""
from backend.emotions import EMOTIONS

# FER-2013 native label order (index 0..6) as distributed.
FER_LABELS: list[str] = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# FER-2013 label name -> our canonical 7-label name.
FER_TO_CANONICAL: dict[str, str] = {
    "angry": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "sad": "sad",
    "surprise": "surprise",
    "neutral": "neutral",
}

assert set(FER_TO_CANONICAL) == set(FER_LABELS)
assert all(v in EMOTIONS for v in FER_TO_CANONICAL.values())


def canonical_index_for_fer_name(name: str) -> int:
    """Map a FER-2013 class NAME to the canonical EMOTIONS index.

    Name-based on purpose: HF mirrors of FER-2013 use different *index* orders
    (e.g. some put neutral at 4, sad at 5), so mapping by integer position is
    unsafe. Callers should read the dataset's own `ClassLabel.names` and pass the
    name through here.
    """
    return EMOTIONS.index(FER_TO_CANONICAL[name])
