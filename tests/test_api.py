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
