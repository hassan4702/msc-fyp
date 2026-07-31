"""Face detection + canonical crop, shared by training and inference.

This module exists for one reason: **training and inference must frame the face
identically**. The previous pipeline trained on raw FER-2013 thumbnails but ran
inference on a Haar bounding box (~0.807x that framing), which cost 6pp of accuracy
without any test catching it. Running the *same* detector and the *same* margin on
both sides makes that class of bug structurally impossible.

MediaPipe BlazeFace replaces the Haar cascade. Measured detection coverage:

    FER-2013 thumbnails   Haar  24.7%   BlazeFace 100.0%
    MELD video frames     Haar  90.0%   BlazeFace  96.3%
    MELD, 30deg head tilt Haar  52.7%   BlazeFace  98.7%

Model file: models/weights/mediapipe/blaze_face_short_range.tflite.
"""
from __future__ import annotations

import os

# Fraction of the detected box width/height added on each side. BlazeFace returns a
# tight box; FER-2013's own framing includes forehead and chin, so we widen to match.
MARGIN = 0.25

_DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "weights", "mediapipe", "blaze_face_short_range.tflite",
)


class FaceDetector:
    """BlazeFace wrapper returning the largest face as a margin-expanded crop.

    Not thread-safe (the underlying graph holds per-call state), so callers that
    share one instance across threads must hold `lock`.
    """

    def __init__(self, model_path: str | None = None, min_confidence: float = 0.2):
        import threading

        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision

        path = model_path or _DEFAULT_MODEL
        if not os.path.isfile(path):
            raise FileNotFoundError(f"BlazeFace model missing: {path}")
        self._mp = mp
        self._detector = vision.FaceDetector.create_from_options(
            vision.FaceDetectorOptions(
                base_options=mpp.BaseOptions(model_asset_path=path),
                min_detection_confidence=min_confidence,
            )
        )
        self.lock = threading.Lock()

    def crop(self, image) -> "object | None":
        """Detect the largest face and return its margin-expanded crop, in grayscale.

        Thin wrapper over `crop_and_box` for the callers that only want pixels.

        Pass a **BGR colour** frame whenever one is available. BlazeFace is trained on
        colour and loses a lot without it — measured on 300 MELD frames, detection
        coverage is 96.3% on colour input against 73.7% on the same frames converted
        to grayscale. A 2-D grayscale array is still accepted (FER-2013 thumbnails have
        no colour to give, and score 100% regardless).

        Returns None when no face is found. The crop is clamped to the image, so a face
        at the frame edge yields a smaller — never an empty or wrapped — array.
        """
        found = self.crop_and_box(image)
        return found[0] if found else None

    def crop_and_box(self, image) -> "tuple | None":
        """`crop`, plus the (x0, y0, x1, y1) box it came from, in image coordinates.

        The box is what the UI draws over the live video so a user can see which face
        the classifier actually read. Detection still runs exactly once per call.
        """
        import cv2
        import numpy as np

        colour = image.ndim == 3
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB if colour else cv2.COLOR_GRAY2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        with self.lock:
            result = self._detector.detect(mp_image)
        if not result.detections:
            return None
        box = max((d.bounding_box for d in result.detections), key=lambda b: b.width * b.height)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if colour else image
        h, w = gray.shape[:2]
        mx, my = int(box.width * MARGIN), int(box.height * MARGIN)
        x0 = max(0, box.origin_x - mx)
        y0 = max(0, box.origin_y - my)
        x1 = min(w, box.origin_x + box.width + mx)
        y1 = min(h, box.origin_y + box.height + my)
        if x1 <= x0 or y1 <= y0:
            return None
        return gray[y0:y1, x0:x1], (x0, y0, x1, y1)
