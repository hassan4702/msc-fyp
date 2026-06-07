"""The single canonical emotion label space used by every component.

We use Ekman's 6 basic emotions + neutral. This is the intersection point that
lets the face model (FER-2013 labels) and the text model (GoEmotions mapped via
the official Ekman grouping) speak the same language, which is what makes late
fusion possible at all.
"""
from enum import Enum


class Emotion(str, Enum):
    ANGER = "anger"
    DISGUST = "disgust"
    FEAR = "fear"
    HAPPY = "happy"
    SAD = "sad"
    SURPRISE = "surprise"
    NEUTRAL = "neutral"


# Canonical ordering for probability vectors. Do NOT reorder — fusion math and
# any saved model classification heads assume this exact order.
EMOTIONS: list[str] = [e.value for e in Emotion]
