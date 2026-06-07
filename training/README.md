# Training

Offline model training — **run on the M4 Max while it is available**, save the
weights, and load them from the backend at inference time. Nothing here runs on
the weak deployment device.

## Text model (Step 1 — implemented)

```bash
pip install -r requirements.txt
python training/train_text.py --output models/weights/text --epochs 3
# then serve the backend using the trained model:
TEXT_MODEL_DIR=models/weights/text uvicorn backend.app:app
```

`goemotions.py` holds the official Ekman 27→7 mapping (unit-tested);
`train_text.py` fine-tunes DistilBERT with class-weighted loss and reports
macro-F1 on the GoEmotions test split.

Other planned scripts (see `docs/plan`):

- `train_face.py` — fine-tune a MobileNet/ResNet backbone on FER-2013
  (optionally AffectNet), with augmentation + class weighting.
- `calibrate.py` — temperature-scale each model so confidences are comparable
  before they reach the fusion layer.

Weights are written to `models/weights/` (git-ignored).
