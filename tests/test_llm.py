from backend.models.llm import REFUSAL, build_system_prompt, enforce_guardrails


def test_guardrail_replaces_code_with_refusal():
    assert enforce_guardrails("sure, here: ```print(1)```") == REFUSAL


def test_guardrail_passes_normal_reply():
    assert enforce_guardrails("It sounds like you're tired.") == "It sounds like you're tired."


def test_system_prompt_carries_rules_and_emotion():
    p = build_system_prompt("sad", conflicted=False)
    assert "ABSOLUTE RULES" in p
    assert "sad" in p


def test_pick_ollama_model(monkeypatch):
    import backend.models.llm as llm

    monkeypatch.setattr(llm, "ollama_models", lambda url: ["nomic-embed-text:latest", "llama3:8b", "qwen2.5:7b"])
    assert llm.pick_ollama_model("x", "qwen2.5:7b") == "qwen2.5:7b"  # preferred is installed
    assert llm.pick_ollama_model("x", "mistral") == "llama3:8b"      # not installed -> first non-embedding
    monkeypatch.setattr(llm, "ollama_models", lambda url: ["mxbai-embed-large:latest"])
    assert llm.pick_ollama_model("x", "") == ""                      # only embedding models
    monkeypatch.setattr(llm, "ollama_models", lambda url: [])
    assert llm.pick_ollama_model("x", "") == ""                      # ollama down


def test_pinned_model_is_never_silently_substituted(monkeypatch):
    """OLLAMA_MODEL set = the user named a model. Serving a different one is a lie.

    The real case: default qwen3:4b with only qwen3:8b pulled. Unpinned, the base-name
    fallback runs 8b -- larger than the model asked for, visible only in GET /health.
    """
    import backend.models.llm as llm

    monkeypatch.setattr(llm, "ollama_models", lambda url: ["qwen3:8b", "llama3:8b"])
    assert llm.pick_ollama_model("x", "qwen3:4b") == "qwen3:8b"                 # unpinned: substitutes
    assert llm.pick_ollama_model("x", "qwen3:4b", pinned=True) == ""            # pinned: refuses
    assert llm.pick_ollama_model("x", "llama3:8b", pinned=True) == "llama3:8b"  # pinned + pulled: exact
    assert llm.pick_ollama_model("x", "qwen3", pinned=True) == "qwen3:8b"       # bare name -> any tag


def test_ollama_timeout_falls_back_instead_of_raising(monkeypatch):
    """A slow model must degrade to the template reply, not 500 the chat turn.

    Real failure on a CPU-only Windows laptop: httpx.ReadTimeout propagated out of
    pipeline.process and the user got a failed message instead of a slower one.
    """
    import httpx

    import backend.models.llm as llm

    def timeout(*a, **kw):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(httpx, "post", timeout)
    reply = llm.OllamaResponder("qwen3:4b").generate("i feel awful", "sad", False)
    assert reply and "sad" in reply          # a real, usable reply came back
    assert "Traceback" not in reply


def test_gemini_network_failure_falls_back_instead_of_raising(monkeypatch):
    import httpx

    import backend.models.llm as llm

    def boom(*a, **kw):
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(httpx, "post", boom)
    assert llm.GeminiResponder("key").generate("hello", "happy", False)


def test_ollama_disables_thinking_and_retries_without_it_on_reject(monkeypatch):
    """Thinking traces are the difference between a reply and a timeout on slow hardware.

    Measured on qwen3:8b: 406 eval tokens / 13.2s with thinking on, 20 / 0.5s with it off.
    Older Ollama builds reject the field, so a 400 must retry rather than lose the LLM.
    """
    import httpx

    import backend.models.llm as llm

    sent = []

    class Resp:
        def __init__(self, code):
            self.status_code = code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError("boom", request=None, response=None)

        def json(self):
            return {"message": {"content": "I hear you."}}

    def post(url, json=None, timeout=None):
        sent.append(dict(json))  # snapshot: the caller reuses and mutates the same dict
        return Resp(400 if len(sent) == 1 else 200)  # first call rejected, retry accepted

    monkeypatch.setattr(httpx, "post", post)
    assert llm.OllamaResponder("qwen3:4b").generate("hi", "sad", False) == "I hear you."
    assert sent[0]["think"] is False        # thinking disabled on the first attempt
    assert "think" not in sent[1]           # dropped on the retry
    assert sent[0]["options"]["num_predict"] == 512
