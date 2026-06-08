"""MELD multimodal evaluation — RQ1 (fusion vs single channels) and RQ2 (conflict).

Pipeline per utterance: read the middle video frame -> face model; transcript ->
text model; then score text-only / face-only / each fusion strategy / majority,
both overall and on the conflict subset (where the two channels disagree). Also
trains the learned arbiter (RQ2) on the train split.

Run on the M4 Max after MELD_raw.tar.gz is downloaded:

    python evaluation/evaluate_meld.py --tar ~/.cache/huggingface/.../MELD_raw.tar.gz \
        --text-model-dir models/weights/text --face-model-path models/weights/face/face_net.pt

Heavy + slow (video decode + two model forwards per utterance); use --limit for a
quick pass. The FER->TV domain gap means the face channel is expected to be weak;
the honest finding is whatever fusion does relative to text-only here.
"""
from __future__ import annotations

import argparse
import base64
import csv
import sys
import tarfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.emotions import EMOTIONS  # noqa: E402
from backend.models.fusion import (  # noqa: E402
    ConfidenceGatedFusion,
    LearnedFusion,
    WeightedAverageFusion,
)
from evaluation.meld import meld_to_canonical  # noqa: E402
from evaluation.scoring import macro_f1  # noqa: E402

# split -> (annotation CSV, candidate video subdir names in the extracted tree)
SPLITS = {
    "train": ("train_sent_emo.csv", ["train_splits", "train"]),
    "dev": ("dev_sent_emo.csv", ["dev_splits_complete", "dev"]),
    "test": ("test_sent_emo.csv", ["output_repeated_splits_test", "test"]),
}


def extract_all(tar_path: str, dest: str) -> Path:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if not any(dest.rglob("*_sent_emo.csv")):
        print(f"extracting {tar_path} -> {dest}")
        with tarfile.open(tar_path) as t:
            t.extractall(dest)
        for nested in list(dest.rglob("*.tar.gz")) + list(dest.rglob("*.tar")):
            with tarfile.open(nested) as t:
                t.extractall(nested.parent)
    return dest


def _find_dir(root: Path, names: list[str]) -> Path | None:
    for name in names:
        for d in root.rglob(name):
            if d.is_dir():
                return d
    return None


def load_split_rows(root: Path, split: str, limit: int = 0) -> list[dict]:
    csv_name, dir_names = SPLITS[split]
    csv_path = next(iter(root.rglob(csv_name)), None)
    video_dir = _find_dir(root, dir_names)
    if csv_path is None or video_dir is None:
        raise FileNotFoundError(f"{split}: csv={csv_path} video_dir={video_dir} under {root}")
    rows = []
    with open(csv_path, newline="", encoding="utf-8", errors="ignore") as f:
        for r in csv.DictReader(f):
            video = video_dir / f"dia{r['Dialogue_ID']}_utt{r['Utterance_ID']}.mp4"
            rows.append({"text": r["Utterance"], "gold": meld_to_canonical(r["Emotion"]), "video": video})
    return rows[:limit] if limit else rows


def middle_frame_b64(video_path: Path, cv2) -> str | None:
    if not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if n > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame)
    return base64.b64encode(buf).decode() if ok else None


def build_records(rows: list[dict], text_model, face_model, cv2) -> list[dict]:
    records = []
    for i, row in enumerate(rows):
        b64 = middle_frame_b64(row["video"], cv2)
        face_pred = face_model.predict([b64] if b64 else [])
        text_pred = text_model.predict(row["text"])
        records.append({"gold": row["gold"], "text": text_pred, "face": face_pred})
        if (i + 1) % 200 == 0:
            print(f"  built {i + 1}/{len(rows)} records")
    return records


class _ArbiterModel:
    """Wraps a fitted sklearn classifier to emit 7-dim probs in EMOTIONS order."""

    def __init__(self, clf):
        self.clf = clf

    def predict_proba(self, rows):
        raw = self.clf.predict_proba(rows)
        out = []
        for row in raw:
            vec = [0.0] * len(EMOTIONS)
            for cls, prob in zip(self.clf.classes_, row):
                vec[int(cls)] = float(prob)
            out.append(vec)
        return out


def train_arbiter(records: list[dict]):
    from sklearn.linear_model import LogisticRegression

    x = [LearnedFusion.features(r["text"], r["face"]) for r in records]
    y = [EMOTIONS.index(r["gold"]) for r in records]
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(x, y)
    return _ArbiterModel(clf)


def evaluate(records: list[dict], strategies: dict) -> dict[str, float]:
    gold = [r["gold"] for r in records]
    majority = Counter(gold).most_common(1)[0][0]
    results = {
        "text_only": macro_f1(gold, [r["text"].label for r in records], EMOTIONS),
        "face_only": macro_f1(gold, [r["face"].label for r in records], EMOTIONS),
        "majority": macro_f1(gold, [majority] * len(records), EMOTIONS),
    }
    for name, fusion in strategies.items():
        preds = [fusion.fuse(r["text"], r["face"]).prediction.label for r in records]
        results[f"fused:{name}"] = macro_f1(gold, preds, EMOTIONS)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar", required=True, help="path to MELD_raw.tar.gz")
    parser.add_argument("--data-dir", default="data/meld")
    parser.add_argument("--text-model-dir", default="models/weights/text")
    parser.add_argument("--face-model-path", default="models/weights/face/face_net.pt")
    parser.add_argument("--limit", type=int, default=0, help="cap utterances per split (0 = all)")
    args = parser.parse_args()

    import cv2

    from backend.models.face_model import CnnFaceEmotionModel
    from backend.models.text_model import TransformerTextEmotionModel

    root = extract_all(args.tar, args.data_dir)
    text_model = TransformerTextEmotionModel(args.text_model_dir)
    face_model = CnnFaceEmotionModel(args.face_model_path)

    print("building train records (for the arbiter)...")
    train_rows = load_split_rows(root, "train", args.limit)
    train_records = build_records(train_rows, text_model, face_model, cv2)
    print("building test records...")
    test_rows = load_split_rows(root, "test", args.limit)
    test_records = build_records(test_rows, text_model, face_model, cv2)

    coverage = sum(1 for r in test_records if r["face"].available) / max(len(test_records), 1)
    print(f"\nface detection coverage on test: {coverage:.1%}")

    arbiter = train_arbiter(train_records)
    strategies = {
        "weighted_avg": WeightedAverageFusion(),
        "confidence_gated": ConfidenceGatedFusion(),
        "learned": LearnedFusion(arbiter),
    }

    print("\n=== RQ1: macro-F1 on MELD test (all utterances) ===")
    for name, score in evaluate(test_records, strategies).items():
        print(f"  {name:22} {score:.4f}")

    conflict = [r for r in test_records if r["face"].available and r["text"].label != r["face"].label]
    print(f"\n=== RQ2: conflict subset ({len(conflict)}/{len(test_records)} utterances) ===")
    if conflict:
        for name, score in evaluate(conflict, strategies).items():
            print(f"  {name:22} {score:.4f}")


if __name__ == "__main__":
    main()
