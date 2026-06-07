"""FastAPI application wiring the pipeline together.

Run locally with:  uvicorn backend.app:app --reload
"""
from fastapi import FastAPI

from backend.config import settings
from backend.models.face_model import StubFaceEmotionModel
from backend.models.fusion import ConfidenceGatedFusion, WeightedAverageFusion
from backend.models.llm import OllamaResponder, TemplateResponder
from backend.models.text_model import StubTextEmotionModel
from backend.schemas import ChatRequest, ChatResponse
from backend.services.pipeline import EmotionPipeline


def build_pipeline() -> EmotionPipeline:
    """Assemble the pipeline from config. Swap stubs for trained models here."""
    text_model = StubTextEmotionModel()
    face_model = StubFaceEmotionModel()

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
