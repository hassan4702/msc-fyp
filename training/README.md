# Training

Offline model training — **run on the M4 Max while it is available**, save the
weights, and load them from the backend at inference time. Nothing here runs on
the weak deployment device.

Planned scripts (see `docs/plan`):

- `train_text.py` — fine-tune DistilBERT on GoEmotions (mapped to the 7 Ekman
  labels), early-stop on validation macro-F1, export the weights.
- `train_face.py` — fine-tune a MobileNet/ResNet backbone on FER-2013
  (optionally AffectNet), with augmentation + class weighting.
- `calibrate.py` — temperature-scale each model so confidences are comparable
  before they reach the fusion layer.

Weights are written to `models/weights/` (git-ignored).
