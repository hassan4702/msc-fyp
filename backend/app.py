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
    pick_ollama_model,
)
from backend.models.text_model import StubTextEmotionModel
from backend.schemas import ChatRequest, ChatResponse, FaceRequest, FaceResponse
from backend.services.pipeline import EmotionPipeline, _view


def _path(p: str) -> str:
    """Resolve a configured path against the repo root, so runs are cwd-independent."""
    return str(Path(__file__).resolve().parent.parent / p) if p and not os.path.isabs(p) else p


def _load_text_model() -> EmotionModel:
    """Use the fine-tuned DistilBERT if TEXT_MODEL_DIR is set; otherwise the stub."""
    if settings.text_model_dir:
        try:
            from backend.models.text_model import TransformerTextEmotionModel

            return TransformerTextEmotionModel(_path(settings.text_model_dir))
        except Exception as exc:  # missing torch/weights -> stay usable
            print(f"[warn] falling back to stub text model: {exc}")
    return StubTextEmotionModel()


def _load_face_model() -> EmotionModel:
    """Use the trained FER CNN if FACE_MODEL_PATH is set; otherwise the stub.

    No isfile() pre-check: a missing file must reach the `except` and print, or a
    mis-set path silently serves the stub's constant "neutral 60%" forever.
    """
    if settings.face_model_path:
        try:
            from backend.models.face_model import CnnFaceEmotionModel

            return CnnFaceEmotionModel(_path(settings.face_model_path))
        except Exception as exc:  # missing torch/cv2/weights -> stay usable
            print(f"[warn] falling back to stub face model: {exc}")
    return StubFaceEmotionModel()


def _pick_responder() -> Responder:
    """auto: Ollama with ANY pulled chat model, else Gemini, else offline template."""
    backend = settings.llm_backend
    if backend == "template":
        return TemplateResponder()
    if backend == "gemini" and settings.gemini_api_key:
        return GeminiResponder(settings.gemini_api_key, settings.gemini_model)
    if backend in ("auto", "ollama"):
        model = pick_ollama_model(settings.ollama_url, settings.ollama_model, settings.ollama_model_pinned)
        if model:
            return OllamaResponder(model, settings.ollama_url)
        if settings.ollama_model_pinned:
            print(f"[warn] OLLAMA_MODEL={settings.ollama_model!r} is not pulled -- "
                  f"run `ollama pull {settings.ollama_model}`. Falling back to Gemini/template.")
    if backend in ("auto", "gemini") and settings.gemini_api_key:
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
    # Pay the model load + system-prompt eval now, in the background, instead of charging it
    # to whoever sends the first message. On a CPU-only laptop that cold pass alone exceeded
    # the request timeout, while every subsequent reply was fine.
    if isinstance(pipeline.responder, OllamaResponder):
        pipeline.responder.warm()

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
            "model": getattr(pipeline.responder, "model", None),
            "face": type(pipeline.face_model).__name__,  # "Stub..." here means no real face model
            "text": type(pipeline.text_model).__name__,
        }

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest):
        return pipeline.process(req.message, req.frames, req.history)

    @app.post("/face", response_model=FaceResponse)
    def face(req: FaceRequest):
        """Face channel on its own, for the live mood ring.

        Deliberately skips the text model and the responder: the UI polls this about once
        a second, and /chat cannot be polled because it runs a 7B LLM per call.
        """
        model = pipeline.face_model
        # The stub has no detector, so no box -- the overlay just stays hidden.
        predict_with_box = getattr(model, "predict_with_box", None)
        pred, box = predict_with_box(req.frames) if predict_with_box else (model.predict(req.frames), None)
        return {"emotion": _view(pred), "box": list(box) if box else None}

    return app


app = create_app()
