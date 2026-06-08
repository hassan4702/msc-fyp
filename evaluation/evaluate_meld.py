"""MELD multimodal evaluation — RQ1 (fusion vs single channels) and RQ2 (conflict).

Pipeline per utterance: read the middle video frame -> face model; transcript ->
text model; then score text-only / face-only / each fusion strategy / majority,
both overall and on the conflict subset (where the two channels disagree). Also
trains the learned arbiter (RQ2) on the train split.

Run on the M4 Max (videos extracted to data/meld/MELD_raw/videos, emotion-label
CSVs in data/meld/labels):

    python evaluation/evaluate_meld.py --text-model-dir models/weights/text \
        --face-model-path models/weights/face/face_net.pt

Heavy + slow (video decode + two model forwards per utterance); use --limit for a
quick pass. The FER->TV domain gap means the face channel is expected to be weak;
the honest finding is whatever fusion does relative to text-only here.
"""
from __future__ import annotations

import argparse
import base64
import csv
import sys
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


def load_split_rows(labels_dir: str, videos_dir: str, split: str, limit: int = 0) -> list[dict]:
    """Join MELD emotion labels to the repackaged video files.

    Videos are named {split}_dia{D}_utt{U}.mp4; the emotion labels come from the
    authoritative declare-lab sent_emo CSVs (the video repackaging dropped the
    emotion column), matched on (split, Dialogue_ID, Utterance_ID).
    """
    csv_path = Path(labels_dir) / f"{split}_sent_emo.csv"
    videos = Path(videos_dir)
    rows = []
    with open(csv_path, newline="", encoding="utf-8", errors="ignore") as f:
        for r in csv.DictReader(f):
            d, u = r["Dialogue_ID"], r["Utterance_ID"]
            rows.append({
                "text": r["Utterance"],
                "gold": meld_to_canonical(r["Emotion"]),
                "video": videos / f"{split}_dia{d}_utt{u}.mp4",
            })
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
    parser.add_argument("--videos-dir", default="data/meld/MELD_raw/videos")
    parser.add_argument("--labels-dir", default="data/meld/labels")
    parser.add_argument("--text-model-dir", default="models/weights/text")
    parser.add_argument("--face-model-path", default="models/weights/face/face_net.pt")
    parser.add_argument("--limit", type=int, default=0, help="cap utterances per split (0 = all)")
    args = parser.parse_args()

    import cv2

    from backend.models.face_model import CnnFaceEmotionModel
    from backend.models.text_model import TransformerTextEmotionModel

    text_model = TransformerTextEmotionModel(args.text_model_dir)
    face_model = CnnFaceEmotionModel(args.face_model_path)

    print("building train records (for the arbiter)...")
    train_rows = load_split_rows(args.labels_dir, args.videos_dir, "train", args.limit)
    train_records = build_records(train_rows, text_model, face_model, cv2)
    print("building test records...")
    test_rows = load_split_rows(args.labels_dir, args.videos_dir, "test", args.limit)
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
