"""API request/response schemas (Pydantic v2)."""
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str
    frames: list[str] = Field(
        default_factory=list,
        max_length=8,  # the UI sends 3; cap it so a client can't post an unbounded batch
        description="Base64-encoded webcam frames; empty = text-only",
    )
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


class FaceRequest(BaseModel):
    frames: list[str] = Field(default_factory=list, max_length=4, description="Base64 webcam frames")


class FaceResponse(BaseModel):
    """Face channel only. Polled every few seconds by the live mood ring, so it must stay LLM-free."""

    emotion: EmotionView
    box: list[int] | None = Field(default=None, description="[x0,y0,x1,y1] of the read face, in frame pixels")
