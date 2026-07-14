"""Runtime configuration via environment variables (with safe defaults).

`LLM_BACKEND=auto` (default) picks the responder at startup: Ollama if it's
running with the model, else Gemini if GEMINI_API_KEY is set, else the offline
template. A repo-root `.env` is loaded automatically (git-ignored).
"""
import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv() -> None:
    """Load repo-root .env into the environment (no dependency, never overrides)."""
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            s = line.strip()
            if s and not s.startswith("#") and "=" in s:
                k, v = s.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()


@dataclass
class Settings:
    llm_backend: str = os.environ.get("LLM_BACKEND", "auto")  # auto | ollama | gemini | template
    ollama_model: str = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
    ollama_url: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    gemini_api_key: str = os.environ.get("GEMINI_API_KEY", "")
    gemini_model: str = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
    text_model_dir: str = os.environ.get("TEXT_MODEL_DIR", "")  # empty -> keyword stub
    face_model_path: str = os.environ.get("FACE_MODEL_PATH", "")  # empty -> stub
    fusion_strategy: str = os.environ.get("FUSION_STRATEGY", "confidence_gated")  # weighted_average | confidence_gated
    text_weight: float = float(os.environ.get("TEXT_WEIGHT", "0.5"))
    conflict_threshold: float = float(os.environ.get("CONFLICT_THRESHOLD", "0.5"))


settings = Settings()
