"""Core contracts shared across the emotion pipeline.

Everything in the system speaks in terms of these types. Both emotion models
(face and text) emit an `EmotionPrediction` over the SAME 7-label space defined
in `backend.emotions`; a `FusionStrategy` combines two predictions into one; a
`Responder` turns the fused emotion into a reply.

Locking these contracts first is what lets the subsystems be built and tested
independently and swapped later (stub -> real model) without touching callers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from backend.emotions import EMOTIONS


@dataclass
class EmotionPrediction:
    """A probability distribution over the 7 canonical emotions."""

    probabilities: dict[str, float]
    label: str
    confidence: float
    source: str  # "text" | "face" | "fused"
    available: bool = True  # False when the modality produced nothing (e.g. no face detected)

    def vector(self) -> list[float]:
        """Probabilities in canonical EMOTIONS order."""
        return [self.probabilities[e] for e in EMOTIONS]

    @classmethod
    def from_scores(
        cls, scores: dict[str, float], source: str, available: bool = True
    ) -> "EmotionPrediction":
        """Build a normalised prediction from (possibly unnormalised) per-emotion scores."""
        total = sum(max(scores.get(e, 0.0), 0.0) for e in EMOTIONS)
        if total <= 0:
            probs = {e: 1.0 / len(EMOTIONS) for e in EMOTIONS}
        else:
            probs = {e: max(scores.get(e, 0.0), 0.0) / total for e in EMOTIONS}
        label = max(probs, key=probs.get)
        return cls(probabilities=probs, label=label, confidence=probs[label], source=source, available=available)

    @classmethod
    def unavailable(cls, source: str) -> "EmotionPrediction":
        """A flat distribution flagged as unavailable (modality contributed nothing)."""
        probs = {e: 1.0 / len(EMOTIONS) for e in EMOTIONS}
        return cls(probabilities=probs, label="neutral", confidence=0.0, source=source, available=False)


class EmotionModel(ABC):
    """A model that maps some input (text string or list of frames) to a prediction."""

    @abstractmethod
    def predict(self, inputs) -> EmotionPrediction: ...


@dataclass
class FusionResult:
    prediction: EmotionPrediction
    conflicted: bool
    text_weight: float
    face_weight: float
    strategy: str


class FusionStrategy(ABC):
    """Combines a text and a face prediction into a single fused prediction."""

    @abstractmethod
    def fuse(self, text: EmotionPrediction, face: EmotionPrediction) -> FusionResult: ...


class Responder(ABC):
    """Generates a reply given the user message and the fused emotional state."""

    @abstractmethod
    def generate(
        self, message: str, emotion: str, conflicted: bool, history: list[dict] | None = None
    ) -> str: ...
