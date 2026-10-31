import numpy as np
import pytest
from fastapi.testclient import TestClient

from core import app as app_module
from core import database, pipeline


class FakeLLM:
    def analyze(self, text, context=None):
        lower = text.lower()
        when = next((d for d in ("friday", "monday", "tuesday") if d in lower), None)
        return {"category": "reminder", "intent": "call_person", "summary": text[:60],
                "slots": {"who": "Ravi" if "ravi" in lower else None, "what": None,
                          "when": when.capitalize() if when else None, "where": None},
                "missing": [], "confidence": 0.9}


class BrokenLLM:
    def analyze(self, text, context=None):
        raise ConnectionError("Ollama is not running")


class FakeJudge:
    def judge(self, **kwargs):
        return "CONTRADICT"


def fake_embed(text):
    return np.array([1, 0, 0] if "ravi" in text.lower() else [0, 1, 0], dtype=np.float32)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(pipeline, "embed", fake_embed)
    monkeypatch.setitem(app_module.state, "llm", FakeLLM())
    monkeypatch.setitem(app_module.state, "judge", FakeJudge())
    with TestClient(app_module.app) as c:
        yield c


def post(client, text):
    return client.post("/notes/text", json={"text": text})


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_complete_note_becomes_knowledge(client):
    r = post(client, "Call Ravi on Friday.")
    assert r.status_code == 200
    assert r.json()["stability"] == "stable"
    assert r.json()["fusion"]["decision"] == "new"
    items = client.get("/knowledge").json()
    assert len(items) == 1 and items[0]["content"]["slots"]["when"] == "Friday"


def test_incomplete_note_is_pending(client):
    post(client, "Remind me to call Ravi.")
    pending = client.get("/pending").json()
    assert len(pending) == 1 and pending[0]["missing"] == ["when"]
    assert client.get("/knowledge").json() == []


def test_later_note_completes_pending_fragment(client):
    post(client, "Remind me to call Ravi.")
    r = post(client, "Call Ravi on Friday.")
    assert r.json()["stability"] == "absorbed"
    assert client.get("/pending").json() == []
    assert len(client.get("/knowledge").json()) == 1


def test_explicit_change_creates_version_two(client):
    post(client, "Call Ravi on Friday.")
    r = post(client, "Actually, move the call with Ravi to Monday.")
    assert r.json()["lifecycle"]["label"] == "UPDATE"

    active = client.get("/knowledge").json()
    assert len(active) == 1 and active[0]["version"] == 2
    assert len(client.get("/knowledge?status=all").json()) == 2

    detail = client.get(f"/knowledge/{active[0]['id']}").json()
    assert [v["version"] for v in detail["history"]] == [1, 2]
    assert len(detail["sources"]) == 1


def test_contradiction_goes_to_review_and_can_be_resolved(client):
    post(client, "Call Ravi on Friday.")
    r = post(client, "Call Ravi on Tuesday.")
    assert r.json()["lifecycle"]["label"] == "CONTRADICT"

    queue = client.get("/review").json()
    assert len(queue) == 1 and queue[0]["type"] == "contradict"
    assert queue[0]["context"]["existing"]["slots"]["when"] == "Friday"

    res = client.post(f"/review/{queue[0]['id']}/resolve", json={"action": "update"})
    assert res.status_code == 200
    active = client.get("/knowledge").json()
    assert len(active) == 1 and active[0]["content"]["slots"]["when"] == "Tuesday"
    assert client.get("/review").json() == []


def test_review_error_cases(client):
    post(client, "Call Ravi on Friday.")
    post(client, "Call Ravi on Tuesday.")

    assert client.post("/review/1/resolve", json={"action": "bogus"}).status_code == 400
    assert client.post("/review/999/resolve", json={"action": "update"}).status_code == 404
    assert client.post("/review/1/resolve", json={"action": "keep_old"}).status_code == 200
    assert client.post("/review/1/resolve", json={"action": "update"}).status_code == 409


def test_blank_note_is_rejected(client):
    assert post(client, "").status_code == 422
    assert post(client, "   ").status_code == 400


def test_model_failure_returns_503_and_keeps_note(client, monkeypatch):
    monkeypatch.setitem(app_module.state, "llm", BrokenLLM())
    assert post(client, "Call Ravi on Friday.").status_code == 503
    notes = client.get("/notes").json()
    assert len(notes) == 1 and notes[0]["status"] == "failed"


def test_confirming_a_pending_fragment_adds_it_to_knowledge(client):
    post(client, "Remind me to call Ravi.")
    fragment_id = client.get("/pending").json()[0]["id"]
    assert client.post(f"/pending/{fragment_id}/confirm").status_code == 200
    assert client.get("/pending").json() == []
    assert len(client.get("/knowledge").json()) == 1
    assert client.post(f"/pending/{fragment_id}/confirm").status_code == 404