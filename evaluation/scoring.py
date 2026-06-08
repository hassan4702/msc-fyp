"""Pure scoring helpers for the MELD comparison (no heavy deps, fully testable).

`compare_systems` is the heart of RQ1: it runs the four systems (text-only,
face-only, fused, majority-class) over the same per-utterance records and reports
macro-F1 for each, so "does fusion beat either channel alone?" is a direct read.
"""
from collections import Counter

from backend.emotions import EMOTIONS
from backend.models.base import EmotionPrediction, FusionStrategy


def macro_f1(gold: list[str], preds: list[str], labels: list[str]) -> float:
    """Unweighted mean per-class F1 (matches sklearn macro F1 with zero_division=0)."""
    f1s = []
    for c in labels:
        tp = sum(1 for g, p in zip(gold, preds) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, preds) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold, preds) if g == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def compare_systems(
    records: list[dict], fusion: FusionStrategy, labels: list[str] = EMOTIONS
) -> dict[str, float]:
    """records: [{"gold": str, "text": EmotionPrediction, "face": EmotionPrediction}, ...]"""
    gold = [r["gold"] for r in records]
    majority = Counter(gold).most_common(1)[0][0]
    text = [r["text"].label for r in records]
    face = [r["face"].label for r in records]
    fused = [fusion.fuse(r["text"], r["face"]).prediction.label for r in records]
    return {
        "text_only": macro_f1(gold, text, labels),
        "face_only": macro_f1(gold, face, labels),
        "fused": macro_f1(gold, fused, labels),
        "majority": macro_f1(gold, [majority] * len(records), labels),
    }
