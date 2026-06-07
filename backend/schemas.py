"""API request/response schemas (Pydantic v2)."""
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str
    frames: list[str] = Field(default_factory=list, description="Base64-encoded webcam frames; empty = text-only")
    history: list[dict] = Field(default_factory=list, description="Prior turns: [{role, content}, ...]")


class EmotionView(BaseModel):
    label: str
    confidence: float
    available: bool
    probabilities: dict[str, float]


class ChatResponse(BaseModel):
    reply: str
    conflicted: bool
    fused_emotion: EmotionView
    text_emotion: EmotionView
    face_emotion: EmotionView
