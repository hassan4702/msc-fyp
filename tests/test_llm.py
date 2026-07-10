from backend.models.llm import REFUSAL, build_system_prompt, enforce_guardrails


def test_guardrail_replaces_code_with_refusal():
    assert enforce_guardrails("sure, here: ```print(1)```") == REFUSAL


def test_guardrail_passes_normal_reply():
    assert enforce_guardrails("It sounds like you're tired.") == "It sounds like you're tired."


def test_system_prompt_carries_rules_and_emotion():
    p = build_system_prompt("sad", conflicted=False)
    assert "ABSOLUTE RULES" in p
    assert "sad" in p
