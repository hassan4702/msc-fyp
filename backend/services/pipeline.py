"""Orchestrates the per-message flow: text + face -> fusion -> responder.

This is the online runtime path. It is deliberately model-agnostic: it depends
only on the abstract contracts in `backend.models.base`, so stub models can be
swapped for trained ones with no change here.
"""
from backend.models.base import EmotionModel, EmotionPrediction, FusionStrategy, Responder


def _view(p: EmotionPrediction) -> dict:
    return {
        "label": p.label,
        "confidence": p.confidence,
        "available": p.available,
        "probabilities": p.probabilities,
    }


class EmotionPipeline:
    def __init__(
        self,
        text_model: EmotionModel,
        face_model: EmotionModel,
        fusion: FusionStrategy,
        responder: Responder,
    ):
        self.text_model = text_model
        self.face_model = face_model
        self.fusion = fusion
        self.responder = responder

    def process(self, message: str, frames: list | None = None, history: list | None = None) -> dict:
        text_pred = self.text_model.predict(message)
        face_pred = self.face_model.predict(frames or [])
        result = self.fusion.fuse(text_pred, face_pred)
        reply = self.responder.generate(message, result.prediction.label, result.conflicted, history)
        return {
            "reply": reply,
            "conflicted": result.conflicted,
            "fused_emotion": _view(result.prediction),
            "text_emotion": _view(text_pred),
            "face_emotion": _view(face_pred),
        }
