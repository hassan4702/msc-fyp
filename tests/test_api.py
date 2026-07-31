from fastapi.testclient import TestClient

from backend.app import create_app


def test_index_page_served():
    r = TestClient(create_app()).get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_health():
    client = TestClient(create_app())
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_chat_endpoint():
    client = TestClient(create_app())
    r = client.post("/chat", json={"message": "I feel sad and tired"})
    assert r.status_code == 200
    body = r.json()
    assert body["reply"]
    assert body["text_emotion"]["label"] == "sad"
    assert "fused_emotion" in body


def test_face_endpoint_without_frames_is_unavailable():
    r = TestClient(create_app()).post("/face", json={"frames": []})
    assert r.status_code == 200
    body = r.json()
    assert body["emotion"]["available"] is False
    assert body["box"] is None


def test_face_endpoint_rejects_oversized_batch():
    # The ring polls ~1/s; a client must not be able to queue an unbounded batch per poll.
    r = TestClient(create_app()).post("/face", json={"frames": ["x"] * 9})
    assert r.status_code == 422


def test_face_endpoint_never_runs_the_full_pipeline(monkeypatch):
    """/face must stay LLM-free -- the ring polls it ~1/s, and process() runs a 7B model.

    Booby-trap process(): if /face ever routes through it, this raises instead of quietly
    costing a model call per second per client.
    """
    from backend.services.pipeline import EmotionPipeline

    def explode(*a, **kw):
        raise AssertionError("/face called the full pipeline")

    monkeypatch.setattr(EmotionPipeline, "process", explode)
    r = TestClient(create_app()).post("/face", json={"frames": []})
    assert r.status_code == 200
    assert "reply" not in r.json()
