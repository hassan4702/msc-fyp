from backend.models.face_model import StubFaceEmotionModel
from backend.models.fusion import ConfidenceGatedFusion
from backend.models.llm import TemplateResponder
from backend.models.text_model import StubTextEmotionModel
from backend.services.pipeline import EmotionPipeline


def _pipeline() -> EmotionPipeline:
    return EmotionPipeline(
        StubTextEmotionModel(), StubFaceEmotionModel(), ConfidenceGatedFusion(), TemplateResponder()
    )


def test_pipeline_returns_reply_and_emotions():
    out = _pipeline().process("I am so happy today, thanks!", frames=[])
    assert isinstance(out["reply"], str) and out["reply"]
    assert out["text_emotion"]["label"] == "happy"
    assert out["face_emotion"]["available"] is False  # no frames supplied


def test_text_only_when_no_frames():
    out = _pipeline().process("hello there", frames=[])
    assert out["face_emotion"]["available"] is False
    assert out["fused_emotion"]["label"] == out["text_emotion"]["label"]
