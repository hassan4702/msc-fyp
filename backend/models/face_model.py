"""Face emotion model.

`StubFaceEmotionModel` runs offline (no webcam/weights). `CnnFaceEmotionModel` is
the real model: detect a face in each frame, classify it, and temporally average
the per-frame distributions. Both implement `EmotionModel.predict(frames)`.

`aggregate_frame_logits` (the no-face / temporal-averaging policy) is torch-free
and unit-tested; the heavy detect+forward path lives in the wrapper.
"""
import base64

from backend.emotions import EMOTIONS
from backend.models.base import EmotionModel, EmotionPrediction, read_temperature, scores_from_logits


def aggregate_frame_logits(
    frame_logits: list[list[float] | None], temperature: float = 1.0
) -> EmotionPrediction:
    """Average per-frame (temperature-scaled) softmax over frames that had a face.

    `None` entries are frames where no face was detected. If none had a face, the
    prediction is flagged unavailable so fusion falls back to text.
    """
    valid = [logits for logits in frame_logits if logits is not None]
    if not valid:
        return EmotionPrediction.unavailable(source="face")
    summed = {e: 0.0 for e in EMOTIONS}
    for logits in valid:
        for emotion, prob in scores_from_logits(logits, temperature).items():
            summed[emotion] += prob
    averaged = {e: summed[e] / len(valid) for e in EMOTIONS}
    return EmotionPrediction.from_scores(averaged, source="face")


class StubFaceEmotionModel(EmotionModel):
    """Placeholder. Returns neutral-ish when a frame is present, unavailable otherwise."""

    def predict(self, inputs: list | None) -> EmotionPrediction:
        frames = inputs or []
        if not frames:
            return EmotionPrediction.unavailable(source="face")
        return EmotionPrediction.from_scores({"neutral": 0.6, "happy": 0.4}, source="face")


class CnnFaceEmotionModel(EmotionModel):
    """ResNet-18 FER model over webcam frames. Heavy deps (torch, cv2) imported lazily.

    Detection and framing come from `backend.models.face_detect`, the same module the
    training script uses, so the crop the model is asked to classify is by construction
    the crop it was trained on.
    """

    def __init__(self, model_path: str, device: str | None = None):
        import os

        import cv2
        import torch

        from backend.models.face_detect import FaceDetector
        from backend.models.face_net import build_resnet18

        self._cv2 = cv2
        self._torch = torch
        self.model = build_resnet18(pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        self.temperature = read_temperature(os.path.dirname(model_path))
        self.detector = FaceDetector()  # holds its own lock; /chat runs in a threadpool

    def _logits_for_frame(self, frame_b64: str) -> list[float] | None:
        import numpy as np

        from backend.models.face_net import preprocess_resnet

        try:
            raw = base64.b64decode(frame_b64.split(",")[-1])
            buf = np.frombuffer(raw, dtype=np.uint8)
            # Decode in COLOUR: the detector needs it. Measured on 300 MELD frames,
            # detection coverage is 96.3% on colour vs 73.7% on the same frames in
            # grayscale. The crop handed to the classifier is grayscale either way.
            img = self._cv2.imdecode(buf, self._cv2.IMREAD_COLOR)
            if img is None:
                return None
            crop = self.detector.crop(img)
            if crop is None:
                return None
            tensor = preprocess_resnet(crop).to(self.device)
            with self._torch.no_grad():
                return self.model(tensor)[0].tolist()
        except Exception as exc:
            print(f"[warn] face frame dropped: {exc!r}")  # never crash the request, but never hide it
            return None

    def predict(self, inputs: list | None) -> EmotionPrediction:
        frames = inputs or []
        return aggregate_frame_logits([self._logits_for_frame(f) for f in frames], self.temperature)
