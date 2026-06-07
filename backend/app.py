"""FastAPI application wiring the pipeline together.

Run locally with:  uvicorn backend.app:app --reload
"""
import os

from fastapi import FastAPI

from backend.config import settings
from backend.models.base import EmotionModel
from backend.models.face_model import StubFaceEmotionModel
from backend.models.fusion import ConfidenceGatedFusion, WeightedAverageFusion
from backend.models.llm import OllamaResponder, TemplateResponder
from backend.models.text_model import StubTextEmotionModel
from backend.schemas import ChatRequest, ChatResponse
from backend.services.pipeline import EmotionPipeline


def _load_text_model() -> EmotionModel:
    """Use the fine-tuned DistilBERT if TEXT_MODEL_DIR is set; otherwise the stub."""
    if settings.text_model_dir and os.path.isdir(settings.text_model_dir):
        try:
            from backend.models.text_model import TransformerTextEmotionModel

            return TransformerTextEmotionModel(settings.text_model_dir)
        except Exception as exc:  # missing torch/weights -> stay usable
            print(f"[warn] falling back to stub text model: {exc}")
    return StubTextEmotionModel()


def _load_face_model() -> EmotionModel:
    """Use the trained FER CNN if FACE_MODEL_PATH is set; otherwise the stub."""
    if settings.face_model_path and os.path.isfile(settings.face_model_path):
        try:
            from backend.models.face_model import CnnFaceEmotionModel

            return CnnFaceEmotionModel(settings.face_model_path)
        except Exception as exc:  # missing torch/cv2/weights -> stay usable
            print(f"[warn] falling back to stub face model: {exc}")
    return StubFaceEmotionModel()


def build_pipeline() -> EmotionPipeline:
    """Assemble the pipeline from config. Swap stubs for trained models here."""
    text_model = _load_text_model()
    face_model = _load_face_model()

    if settings.fusion_strategy == "weighted_average":
        fusion = WeightedAverageFusion(settings.text_weight, settings.conflict_threshold)
    else:
        fusion = ConfidenceGatedFusion(settings.conflict_threshold)

    if settings.llm_backend == "ollama":
        responder = OllamaResponder(settings.ollama_model, settings.ollama_url)
    else:
        responder = TemplateResponder()

    return EmotionPipeline(text_model, face_model, fusion, responder)


def create_app() -> FastAPI:
    app = FastAPI(title="Multimodal Emotion-Aware Chatbot", version="0.1.0")
    pipeline = build_pipeline()

    @app.get("/health")
    def health():
        return {"status": "ok", "fusion": settings.fusion_strategy, "llm": settings.llm_backend}

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest):
        return pipeline.process(req.message, req.frames, req.history)

    return app


app = create_app()
