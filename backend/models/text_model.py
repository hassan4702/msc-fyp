"""Text emotion model.

`StubTextEmotionModel` is a keyword heuristic so the skeleton runs end-to-end
before the real model exists. It will be replaced by `BertTextEmotionModel`
(DistilBERT fine-tuned on GoEmotions, mapped to the 7 Ekman labels) in Phase 3.
The real model must implement the same `EmotionModel.predict(str)` contract.
"""
from backend.emotions import EMOTIONS
from backend.models.base import EmotionModel, EmotionPrediction

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
