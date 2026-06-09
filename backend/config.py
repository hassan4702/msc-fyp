"""Runtime configuration via environment variables (with safe defaults).

Defaults run the offline skeleton with no external services. Switch the LLM
backend to "ollama" once a model is pulled on the M4 Max.
"""
import os
from dataclasses import dataclass


@dataclass
class Settings:
    llm_backend: str = os.environ.get("LLM_BACKEND", "template")  # template | ollama
    ollama_model: str = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
    ollama_url: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    text_model_dir: str = os.environ.get("TEXT_MODEL_DIR", "")  # empty -> keyword stub
    face_model_path: str = os.environ.get("FACE_MODEL_PATH", "")  # empty -> stub
    fusion_strategy: str = os.environ.get("FUSION_STRATEGY", "confidence_gated")  # weighted_average | confidence_gated
    text_weight: float = float(os.environ.get("TEXT_WEIGHT", "0.5"))
    conflict_threshold: float = float(os.environ.get("CONFLICT_THRESHOLD", "0.5"))


settings = Settings()
