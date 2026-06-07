"""Face emotion model.

`StubFaceEmotionModel` returns a fixed distribution (or 'unavailable' when no
frame is supplied) so the pipeline runs without a webcam or trained weights. It
will be replaced by `CnnFaceEmotionModel` (a MobileNet/ResNet backbone
fine-tuned on FER-2013, optionally AffectNet) in Phase 2. The real model takes a
list of frames, runs face detection + a short temporal average, and must
implement the same `EmotionModel.predict(frames)` contract.
"""
from backend.models.base import EmotionModel, EmotionPrediction


class StubFaceEmotionModel(EmotionModel):
    """Placeholder. Returns neutral-ish when a frame is present, unavailable otherwise."""

    def predict(self, inputs: list | None) -> EmotionPrediction:
        frames = inputs or []
        if not frames:
            # No frame -> simulates "no face detected"; fusion will fall back to text.
            return EmotionPrediction.unavailable(source="face")
        return EmotionPrediction.from_scores({"neutral": 0.6, "happy": 0.4}, source="face")
