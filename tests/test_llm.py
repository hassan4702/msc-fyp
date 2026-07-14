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
