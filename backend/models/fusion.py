"""Fusion strategies — the core research contribution.

Two strategies ship in the skeleton:
- WeightedAverageFusion: fixed-weight late fusion (the simple baseline, RQ1).
- ConfidenceGatedFusion: weights scale with per-modality confidence and fall
  back to whichever modality is available; flags conflict when the two
  modalities confidently disagree (RQ2).

A learned arbiter (LearnedFusion) trained on MELD is planned for Phase 4 — see
docs/plan. It will implement the same FusionStrategy.fuse() contract.
"""
from backend.emotions import EMOTIONS
from backend.models.base import EmotionPrediction, FusionResult, FusionStrategy


def _combine(text_vec: list[float], face_vec: list[float], w_text: float, w_face: float) -> list[float]:
    total = w_text + w_face
    if total <= 0:
        w_text = w_face = 0.5
        total = 1.0
    return [(w_text * t + w_face * f) / total for t, f in zip(text_vec, face_vec)]


def _is_conflict(text: EmotionPrediction, face: EmotionPrediction, threshold: float) -> bool:
    if not (text.available and face.available):
        return False
    return text.label != face.label and text.confidence >= threshold and face.confidence >= threshold


class WeightedAverageFusion(FusionStrategy):
    """P_fused = w * P_text + (1 - w) * P_face. The simple, tunable baseline."""

    def __init__(self, text_weight: float = 0.5, conflict_threshold: float = 0.5):
        self.text_weight = text_weight
        self.conflict_threshold = conflict_threshold

    def fuse(self, text: EmotionPrediction, face: EmotionPrediction) -> FusionResult:
        if not face.available:
            return FusionResult(text, False, 1.0, 0.0, "weighted_average")
        if not text.available:
            return FusionResult(face, False, 0.0, 1.0, "weighted_average")
        w_text, w_face = self.text_weight, 1.0 - self.text_weight
        vec = _combine(text.vector(), face.vector(), w_text, w_face)
        fused = EmotionPrediction.from_scores(dict(zip(EMOTIONS, vec)), source="fused")
        return FusionResult(fused, _is_conflict(text, face, self.conflict_threshold),
                            w_text, w_face, "weighted_average")


class LearnedFusion(FusionStrategy):
    """Arbiter (RQ2): a trained classifier decides the label from both predictions.

    The model is injected (must expose `predict_proba(rows) -> list[list[float]]`
    over the 7 classes in EMOTIONS order), so this stays sklearn-free and testable.
    It is trained on paired (text, face, gold) data in evaluation/evaluate_meld.py.
    """

    def __init__(self, model, conflict_threshold: float = 0.5):
        self.model = model
        self.conflict_threshold = conflict_threshold

    @staticmethod
    def features(text: EmotionPrediction, face: EmotionPrediction) -> list[float]:
        return text.vector() + face.vector() + [text.confidence, face.confidence, float(face.available)]

    def fuse(self, text: EmotionPrediction, face: EmotionPrediction) -> FusionResult:
        probs = self.model.predict_proba([self.features(text, face)])[0]
        fused = EmotionPrediction.from_scores(dict(zip(EMOTIONS, probs)), source="fused")
        return FusionResult(fused, _is_conflict(text, face, self.conflict_threshold), 0.0, 0.0, "learned")


class ConfidenceGatedFusion(FusionStrategy):
    """Weights scale with each modality's confidence; missing modality is dropped."""

    def __init__(self, conflict_threshold: float = 0.5):
        self.conflict_threshold = conflict_threshold

    def fuse(self, text: EmotionPrediction, face: EmotionPrediction) -> FusionResult:
        if not face.available:
            return FusionResult(text, False, 1.0, 0.0, "confidence_gated")
        if not text.available:
            return FusionResult(face, False, 0.0, 1.0, "confidence_gated")
        w_text, w_face = text.confidence, face.confidence
        vec = _combine(text.vector(), face.vector(), w_text, w_face)
        fused = EmotionPrediction.from_scores(dict(zip(EMOTIONS, vec)), source="fused")
        total = (w_text + w_face) or 1.0
        return FusionResult(fused, _is_conflict(text, face, self.conflict_threshold),
                            w_text / total, w_face / total, "confidence_gated")
