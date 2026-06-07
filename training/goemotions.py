"""GoEmotions (28 labels) -> canonical 7 Ekman labels.

The grouping is the **official Ekman mapping** published by the GoEmotions authors
(google-research/google-research `goemotions/data/ekman_mapping.json`), so the
collapse is citable rather than arbitrary. GoEmotions' own names "joy" and
"sadness" map onto our labels "happy" and "sad".

GoEmotions is multi-label. To produce the single-label, sums-to-one distribution
the fusion layer expects, we keep only examples whose raw labels fall into a
single Ekman bucket (this includes all single-label examples plus multi-label
examples that already agree on one bucket); genuinely cross-bucket examples are
dropped. `train_text.py` logs the dropped fraction.
"""
from backend.emotions import EMOTIONS

# GoEmotions label index order (0..27), from the "simplified" config.
GOEMOTIONS_LABELS: list[str] = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust",
    "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love",
    "nervousness", "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]

# Official Ekman grouping. Keys use GoEmotions' names ("joy", "sadness"); we
# translate those to our canonical labels ("happy", "sad") when building the map.
_EKMAN_GROUPS: dict[str, list[str]] = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": ["joy", "amusement", "approval", "excitement", "gratitude", "love",
            "optimism", "relief", "pride", "admiration", "desire", "caring"],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
}

# GoEmotions group name -> our canonical 7-label name.
_GROUP_TO_CANONICAL = {"joy": "happy", "sadness": "sad"}


def _build_mapping() -> dict[str, str]:
    mapping: dict[str, str] = {"neutral": "neutral"}
    for group, members in _EKMAN_GROUPS.items():
        canonical = _GROUP_TO_CANONICAL.get(group, group)
        for name in members:
            mapping[name] = canonical
    return mapping


GOEMOTION_TO_EKMAN: dict[str, str] = _build_mapping()

# Sanity: every GoEmotions label has a mapping to one of the 7 canonical labels.
assert set(GOEMOTION_TO_EKMAN) == set(GOEMOTIONS_LABELS)
assert all(v in EMOTIONS for v in GOEMOTION_TO_EKMAN.values())


def ekman_labels_for(indices: list[int]) -> set[str]:
    """The set of canonical labels a GoEmotions example's raw label indices map to."""
    return {GOEMOTION_TO_EKMAN[GOEMOTIONS_LABELS[i]] for i in indices}


def single_ekman_label(indices: list[int]) -> str | None:
    """Return the single canonical label for an example, or None if ambiguous/empty."""
    labels = ekman_labels_for(indices)
    return next(iter(labels)) if len(labels) == 1 else None
