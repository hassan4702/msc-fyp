"""Responder backends.

The responder is the only heavy component, so it sits behind one interface with
swappable backends:
- TemplateResponder: offline, deterministic; default for the skeleton and tests.
- OllamaResponder: local quantized LLM via Ollama (development on the M4 Max).
- ApiResponder (hosted LLM API for the weak deployment device) is added in Phase 4.

Privacy: only the emotion LABEL and the user's TEXT ever reach the responder —
never webcam frames. This matches the GDPR commitments in the risk assessment.
"""
from backend.models.base import Responder

SYSTEM_TEMPLATE = (
    "You are Empath, a warm, emotionally attuned companion. A separate system has "
    "detected that the user seems to be feeling {emotion}. In your reply: (1) gently "
    'name the emotion you sense — e.g. "it sounds like you might be feeling {emotion}" '
    "— and ask whether that's really how they feel, so they can correct you; then "
    "(2) respond supportively to what they actually said. Be warm and natural, never "
    "clinical or repetitive.{conflict}"
)
CONFLICT_NOTE = (
    " Their words and their facial expression seem to point to different feelings, so "
    "acknowledge that gently and ask which is closer to the truth."
)


def build_system_prompt(emotion: str, conflicted: bool) -> str:
    return SYSTEM_TEMPLATE.format(emotion=emotion, conflict=CONFLICT_NOTE if conflicted else "")


class TemplateResponder(Responder):
    """Offline placeholder reply. Lets the whole system run with no LLM installed."""

    def generate(self, message: str, emotion: str, conflicted: bool, history=None) -> str:
        if conflicted:
            return (
                "I want to check in — your words and your expression seem to point in "
                "different directions. How are you really feeling right now?"
            )
        return (
            f"It sounds like you might be feeling {emotion} — is that right? "
            "I'm here either way; tell me more."
        )


class OllamaResponder(Responder):
    """Local LLM via Ollama. Requires the `ollama` service running with the model pulled."""

    def __init__(self, model: str = "mistral", url: str = "http://localhost:11434"):
        self.model = model
        self.url = url

    def generate(self, message: str, emotion: str, conflicted: bool, history=None) -> str:
        import httpx

        messages = [{"role": "system", "content": build_system_prompt(emotion, conflicted)}]
        messages += history or []
        messages.append({"role": "user", "content": message})
        resp = httpx.post(
            f"{self.url}/api/chat",
            json={"model": self.model, "messages": messages, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
