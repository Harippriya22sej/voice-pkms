import numpy as np
import pytest
from fastapi.testclient import TestClient

from core import app as app_module
from core import database, pipeline, retrieval


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


class FakeSpeech:
    def __init__(self, text="Call Ravi on Friday."):
        self.text = text

    def transcribe(self, audio_path):
        return self.text


def fake_embed(text):
    return np.array([1, 0, 0] if "ravi" in text.lower() else [0, 1, 0], dtype=np.float32)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(app_module, "AUDIO_DIR", tmp_path / "audio")
    monkeypatch.setattr(pipeline, "embed", fake_embed)
    monkeypatch.setattr(retrieval, "embed", fake_embed)
    monkeypatch.setitem(app_module.state, "llm", FakeLLM())
    monkeypatch.setitem(app_module.state, "judge", FakeJudge())
    monkeypatch.setitem(app_module.state, "speech", FakeSpeech())
    with TestClient(app_module.app) as c:
        yield c


def post(client, text):
    return client.post("/notes/text", json={"text": text})


def upload(client, name="memo.wav", data=b"fake audio bytes"):
    return client.post("/notes/voice", files={"file": (name, data, "audio/wav")})


def age_all_fragments():
    with database.get_connection() as conn:
        conn.execute("UPDATE fragments SET created_at = '2000-01-01 00:00:00'")


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


def test_voice_note_runs_the_full_pipeline(client):
    r = upload(client)
    assert r.status_code == 200
    body = r.json()
    assert body["transcript"] == "Call Ravi on Friday."
    assert body["stability"] == "stable"
    assert client.get(f"/notes/{body['note_id']}").json()["source"] == "voice"
    assert len(list(app_module.AUDIO_DIR.iterdir())) == 1
    assert len(client.get("/knowledge").json()) == 1


def test_voice_rejects_unsupported_file_type(client):
    assert upload(client, name="notes.txt").status_code == 415


def test_voice_rejects_empty_file(client):
    assert upload(client, data=b"").status_code == 400


def test_voice_rejects_oversized_file(client, monkeypatch):
    monkeypatch.setattr(app_module, "MAX_AUDIO_BYTES", 10)
    assert upload(client, data=b"x" * 100).status_code == 413
    assert list(app_module.AUDIO_DIR.iterdir()) == []   # partial file removed


def test_voice_with_no_speech_returns_422(client, monkeypatch):
    monkeypatch.setitem(app_module.state, "speech", FakeSpeech(""))
    assert upload(client).status_code == 422


def test_search_endpoint_returns_matching_knowledge(client):
    post(client, "Call Ravi on Friday.")
    results = client.get("/search", params={"q": "Ravi call"}).json()
    assert len(results) == 1 and results[0]["slots"]["when"] == "Friday"
    assert client.get("/search", params={"q": "   "}).status_code == 400


def test_category_filter(client):
    post(client, "Call Ravi on Friday.")                  # the fake LLM labels it "reminder"
    assert len(client.get("/knowledge", params={"category": "reminder"}).json()) == 1
    assert client.get("/knowledge", params={"category": "task"}).json() == []
    assert client.get("/knowledge", params={"category": "bogus"}).status_code == 400


def test_stats_counts(client):
    post(client, "Call Ravi on Friday.")
    post(client, "Remind me to call Ravi.")
    assert client.get("/stats").json() == {
        "notes": 2, "pending_fragments": 1, "active_knowledge": 1,
        "superseded_knowledge": 0, "pending_reviews": 0}


def test_stale_pending_is_queued_and_can_be_confirmed(client):
    post(client, "Remind me to call Ravi.")
    age_all_fragments()
    assert client.post("/maintenance/stale-check", params={"hours": 24}).json() == {"queued": 1}

    queue = client.get("/review").json()
    assert len(queue) == 1 and queue[0]["type"] == "stabilize"
    assert queue[0]["context"]["new_note"] == "Remind me to call Ravi."

    res = client.post(f"/review/{queue[0]['id']}/resolve", json={"action": "confirm"})
    assert res.status_code == 200
    assert client.get("/pending").json() == []
    assert len(client.get("/knowledge").json()) == 1


def test_keep_waiting_restarts_the_clock(client):
    post(client, "Remind me to call Ravi.")
    age_all_fragments()
    client.post("/maintenance/stale-check", params={"hours": 24})
    review_id = client.get("/review").json()[0]["id"]

    res = client.post(f"/review/{review_id}/resolve", json={"action": "keep_waiting"})
    assert res.status_code == 200
    assert len(client.get("/pending").json()) == 1
    assert client.post("/maintenance/stale-check", params={"hours": 24}).json() == {"queued": 0}


def test_stale_check_never_queues_the_same_question_twice(client):
    post(client, "Remind me to call Ravi.")
    age_all_fragments()
    assert client.post("/maintenance/stale-check", params={"hours": 24}).json() == {"queued": 1}
    assert client.post("/maintenance/stale-check", params={"hours": 24}).json() == {"queued": 0}
    assert len(client.get("/review").json()) == 1


def test_stale_review_rejects_unknown_action(client):
    post(client, "Remind me to call Ravi.")
    age_all_fragments()
    client.post("/maintenance/stale-check", params={"hours": 24})
    review_id = client.get("/review").json()[0]["id"]
    assert client.post(f"/review/{review_id}/resolve", json={"action": "bogus"}).status_code == 400

def test_ui_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Voice Knowledge" in r.text