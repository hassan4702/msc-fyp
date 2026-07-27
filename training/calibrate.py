"""Temperature-scale the trained models so their confidences are comparable.

Confidence gating in the fusion layer only makes sense if `conf_text` and
`conf_face` mean the same thing. Neural nets are overconfident, so we fit a single
scalar temperature T per model on its validation split (minimising NLL) and write
it to `<model_dir>/calibration.json`. The wrappers read it automatically.

Run on the M4 Max after training both models:

    python training/calibrate.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.emotions import EMOTIONS  # noqa: E402
from backend.models.base import scores_from_logits  # noqa: E402


def expected_calibration_error(confidences: list[float], correct: list[bool], n_bins: int = 10) -> float:
    """ECE: average |accuracy - confidence| across equal-width confidence bins."""
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for conf, ok in zip(confidences, correct):
        bins[min(int(conf * n_bins), n_bins - 1)].append((conf, ok))
    n = len(confidences) or 1
    ece = 0.0
    for b in bins:
        if not b:
            continue
        avg_conf = sum(c for c, _ in b) / len(b)
        acc = sum(1 for _, ok in b if ok) / len(b)
        ece += (len(b) / n) * abs(acc - avg_conf)
    return ece


def fit_temperature(logits_list: list[list[float]], labels: list[int], max_iter: int = 200) -> float:
    """Fit the single temperature that minimises NLL (LBFGS over log T to keep T > 0)."""
    import torch

    logits = torch.tensor(logits_list, dtype=torch.float32)
    targets = torch.tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(logits / log_t.exp(), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


def _ece_from_logits(logits_list, labels) -> float:
    confidences, correct = [], []
    for logits, label in zip(logits_list, labels):
        probs = scores_from_logits(logits)
        pred = max(range(len(EMOTIONS)), key=lambda i: probs[EMOTIONS[i]])
        confidences.append(probs[EMOTIONS[pred]])
        correct.append(pred == label)
    return expected_calibration_error(confidences, correct)


def _ece_at(logits_list, labels, temperature: float) -> float:
    conf, correct = [], []
    for logits, label in zip(logits_list, labels):
        probs = scores_from_logits(logits, temperature)
        pred = max(range(len(EMOTIONS)), key=lambda i: probs[EMOTIONS[i]])
        conf.append(probs[EMOTIONS[pred]])
        correct.append(pred == label)
    return expected_calibration_error(conf, correct)


def _calibrate(name: str, directory: str, logits_list, labels):
    """Fit T on half the split, then only keep it if it improves ECE on the other half.

    The hold-out check is not ceremony. The validation split is also the split the best
    checkpoint was selected on, so the model is mildly overfit to it and the NLL-optimal
    temperature overshoots: for the face model, the fitted T=1.29 gave ECE 0.069 on
    held-out data where T=1.0 gave 0.028. Fitting alone would have shipped a temperature
    that makes confidences worse — which is exactly what the previous version did.
    """
    mid = len(logits_list) // 2
    fit_x, fit_y = logits_list[:mid], labels[:mid]
    held_x, held_y = logits_list[mid:], labels[mid:]

    temperature = fit_temperature(fit_x, fit_y)
    ece_plain = _ece_at(held_x, held_y, 1.0)
    ece_scaled = _ece_at(held_x, held_y, temperature)
    if ece_scaled >= ece_plain:
        print(f"[{name}] fitted T={temperature:.3f} REJECTED "
              f"(held-out ECE {ece_plain:.4f} at T=1.0 vs {ece_scaled:.4f} scaled) -> keeping T=1.0")
        temperature, ece_scaled = 1.0, ece_plain

    Path(directory, "calibration.json").write_text(json.dumps({"temperature": temperature}, indent=2))
    print(f"[{name}] T={temperature:.3f} | held-out ECE {ece_plain:.4f} -> {ece_scaled:.4f} "
          f"| wrote {directory}/calibration.json")


def _collect_text(model_dir: str):
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from training.goemotions import single_ekman_label

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split="validation")
    logits_list, labels = [], []
    with torch.no_grad():
        for ex in ds:
            label = single_ekman_label(ex["labels"])
            if label is None:
                continue
            enc = tok(ex["text"], return_tensors="pt", truncation=True, max_length=128)
            logits_list.append(model(**enc).logits[0].tolist())
            labels.append(EMOTIONS.index(label))
    return logits_list, labels


def _collect_face(model_path: str):
    """Face logits on the validation split, through the *deployed* preprocessing.

    Critically, this reuses the same BlazeFace crops the model was trained on
    (`training.train_face_resnet._load_crops`). The previous version fitted the
    temperature on raw FER thumbnails while inference ran on Haar boxes, so the
    fitted value described a pipeline that never actually ran.

    Images leaked from the train split are dropped, or the confidences being
    calibrated would partly be confidences on memorised examples.
    """
    import hashlib

    import torch

    from backend.models.face_net import build_resnet18, preprocess_resnet
    from training.train_face_resnet import _load_crops

    model = build_resnet18(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    splits = _load_crops("Aaryan333/fer2013_train_publicTest_privateTest", "data/fer_crops.npz")
    train_hashes = {hashlib.md5(a.tobytes()).hexdigest() for a in splits["train"][0]}
    images, labels_all = splits["val"]
    logits_list, labels = [], []
    with torch.no_grad():
        for img, label in zip(images, labels_all):
            if hashlib.md5(img.tobytes()).hexdigest() in train_hashes:
                continue
            logits_list.append(model(preprocess_resnet(img))[0].tolist())
            labels.append(label)
    return logits_list, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-model-dir", default="models/weights/text")
    parser.add_argument("--face-model-path", default="models/weights/face/resnet18.pt")
    args = parser.parse_args()

    if Path(args.text_model_dir).is_dir():
        logits, labels = _collect_text(args.text_model_dir)
        _calibrate("text", args.text_model_dir, logits, labels)
    if Path(args.face_model_path).is_file():
        logits, labels = _collect_face(args.face_model_path)
        _calibrate("face", str(Path(args.face_model_path).parent), logits, labels)


if __name__ == "__main__":
    main()
