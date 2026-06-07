"""Face emotion model.

`StubFaceEmotionModel` runs offline (no webcam/weights). `CnnFaceEmotionModel` is
the real model: detect a face in each frame, classify it, and temporally average
the per-frame distributions. Both implement `EmotionModel.predict(frames)`.

`aggregate_frame_logits` (the no-face / temporal-averaging policy) is torch-free
and unit-tested; the heavy detect+forward path lives in the wrapper.
"""
import base64

from backend.emotions import EMOTIONS
from backend.models.base import EmotionModel, EmotionPrediction, scores_from_logits


def aggregate_frame_logits(frame_logits: list[list[float] | None]) -> EmotionPrediction:
    """Average per-frame softmax over frames that contained a face.

    `None` entries are frames where no face was detected. If none had a face, the
    prediction is flagged unavailable so fusion falls back to text.
    """
    valid = [logits for logits in frame_logits if logits is not None]
    if not valid:
        return EmotionPrediction.unavailable(source="face")
    summed = {e: 0.0 for e in EMOTIONS}
    for logits in valid:
        for emotion, prob in scores_from_logits(logits).items():
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
    """FER-2013 CNN over webcam frames. Heavy deps (torch, cv2) imported lazily."""

    def __init__(self, model_path: str, device: str | None = None):
        import cv2
        import torch

        from backend.models.face_net import FaceNet

        self._cv2 = cv2
        self._torch = torch
        self.model = FaceNet()
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade)

    def _logits_for_frame(self, frame_b64: str) -> list[float] | None:
        import numpy as np

        from backend.models.face_net import INPUT_SIZE, preprocess_gray

        raw = base64.b64decode(frame_b64.split(",")[-1])
        buf = np.frombuffer(raw, dtype=np.uint8)
        img = self._cv2.imdecode(buf, self._cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        faces = self.detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # largest face
        crop = self._cv2.resize(img[y:y + h, x:x + w], (INPUT_SIZE, INPUT_SIZE))
        tensor = preprocess_gray(crop).to(self.device)
        with self._torch.no_grad():
            return self.model(tensor)[0].tolist()

    def predict(self, inputs: list | None) -> EmotionPrediction:
        frames = inputs or []
        return aggregate_frame_logits([self._logits_for_frame(f) for f in frames])
