"""Text emotion model.

`StubTextEmotionModel` is a keyword heuristic so the skeleton runs end-to-end
before the real model exists. `TransformerTextEmotionModel` is the real model
(DistilBERT fine-tuned on GoEmotions, mapped to the 7 Ekman labels). Both
implement the same `EmotionModel.predict(str)` contract.

The model is trained with label index i == EMOTIONS[i], so output logit i
corresponds to EMOTIONS[i]; `_scores_from_logits` relies on that ordering.
"""
import math

from backend.emotions import EMOTIONS
from backend.models.base import EmotionModel, EmotionPrediction


def _scores_from_logits(logits: list[float]) -> dict[str, float]:
    """Softmax over the 7 logits, keyed by emotion in canonical EMOTIONS order."""
    top = max(logits)
    exps = [math.exp(x - top) for x in logits]
    total = sum(exps)
    return {emotion: e / total for emotion, e in zip(EMOTIONS, exps)}

_KEYWORDS = {
    "anger": ["angry", "furious", "annoyed", "hate", "mad"],
    "disgust": ["disgusting", "gross", "sick", "revolting"],
    "fear": ["scared", "afraid", "worried", "anxious", "nervous"],
    "happy": ["happy", "glad", "great", "love", "awesome", "thanks", "thank"],
    "sad": ["sad", "unhappy", "depressed", "down", "crying", "tired", "lonely"],
    "surprise": ["surprised", "wow", "shocked", "unexpected"],
}


class StubTextEmotionModel(EmotionModel):
    """Keyword-based placeholder. NOT for evaluation — skeleton/demo only."""

    def predict(self, inputs: str) -> EmotionPrediction:
        text = (inputs or "").lower()
        scores = {e: 0.0 for e in EMOTIONS}
        for emotion, words in _KEYWORDS.items():
            for w in words:
                if w in text:
                    scores[emotion] += 1.0
        if sum(scores.values()) == 0:
            scores["neutral"] = 1.0
        return EmotionPrediction.from_scores(scores, source="text")


class TransformerTextEmotionModel(EmotionModel):
    """DistilBERT fine-tuned on GoEmotions (mapped to the 7 Ekman labels).

    Heavy deps (torch, transformers) are imported lazily so the rest of the
    backend runs without them. DistilBERT is light enough for CPU inference on
    the weak deployment device.
    """

    def __init__(self, model_dir: str, device: str | None = None, max_length: int = 128):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        self._torch = torch
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, inputs: str) -> EmotionPrediction:
        enc = self.tokenizer(
            inputs or "", return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(self.device)
        with self._torch.no_grad():
            logits = self.model(**enc).logits[0].tolist()
        return EmotionPrediction.from_scores(_scores_from_logits(logits), source="text")
