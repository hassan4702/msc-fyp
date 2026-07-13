import os

# Tests use the offline template responder unless explicitly overridden, so they
# never hit Ollama or the Gemini API (fast + hermetic). Must run before backend
# imports, which pytest guarantees for a root conftest.
os.environ.setdefault("LLM_BACKEND", "template")
os.environ.setdefault("TEXT_MODEL_DIR", "")  # stub models in tests, not the real weights
os.environ.setdefault("FACE_MODEL_PATH", "")
