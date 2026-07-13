"""FastAPI application wiring the pipeline together.

Run locally with:  uvicorn backend.app:app --reload
"""
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from backend.config import settings
from backend.models.base import EmotionModel
from backend.models.face_model import StubFaceEmotionModel
from backend.models.fusion import ConfidenceGatedFusion, WeightedAverageFusion
from backend.models.base import Responder
from backend.models.llm import (
    GeminiResponder,
    OllamaResponder,
    TemplateResponder,
    ollama_available,
)
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


def _pick_responder() -> Responder:
    """auto: Ollama if running with the model, else Gemini, else offline template."""
    backend = settings.llm_backend
    if backend == "template":
        return TemplateResponder()
    if backend == "ollama":
        return OllamaResponder(settings.ollama_model, settings.ollama_url)
    if backend == "gemini" and settings.gemini_api_key:
        return GeminiResponder(settings.gemini_api_key, settings.gemini_model)
    # auto
    if ollama_available(settings.ollama_url, settings.ollama_model):
        return OllamaResponder(settings.ollama_model, settings.ollama_url)
    if settings.gemini_api_key:
        return GeminiResponder(settings.gemini_api_key, settings.gemini_model)
    return TemplateResponder()


def build_pipeline() -> EmotionPipeline:
    """Assemble the pipeline from config. Swap stubs for trained models here."""
    text_model = _load_text_model()
    face_model = _load_face_model()

    if settings.fusion_strategy == "weighted_average":
        fusion = WeightedAverageFusion(settings.text_weight, settings.conflict_threshold)
    else:
        fusion = ConfidenceGatedFusion(settings.conflict_threshold)

    return EmotionPipeline(text_model, face_model, fusion, _pick_responder())


def create_app() -> FastAPI:
    app = FastAPI(title="Multimodal Emotion-Aware Chatbot", version="0.1.0")
    pipeline = build_pipeline()

    @app.get("/")
    def index():
        return FileResponse(
            Path(__file__).resolve().parent.parent / "frontend" / "index.html",
            headers={"Cache-Control": "no-cache"},  # always serve the latest UI while iterating
        )

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "fusion": settings.fusion_strategy,
            "llm": type(pipeline.responder).__name__,
        }

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest):
        return pipeline.process(req.message, req.frames, req.history)

    return app


app = create_app()
