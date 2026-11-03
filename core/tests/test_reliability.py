import json

import numpy as np
import pytest

from core import database
from core.embeddings import to_blob
from core.fusion import fuse_fragment
from core.intents import validate_proposal
from core.lifecycle import resolve_conflict
from core.llm_adapter import OllamaLLMAdapter
from core.stabilization import stabilize

V1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
V_SIMILAR = np.array([0.9, 0.1, 0.0], dtype=np.float32)


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    database.init_db()


def make_note(text="x"):
    with database.get_connection() as conn:
        return conn.execute("INSERT INTO notes (text) VALUES (?)", (text,)).lastrowid


def proposal(intent="call_person", who="Ravi", when=None, what=None):
    return {"category": "reminder", "intent": intent, "summary": "s",
            "slots": {"who": who, "what": what, "when": when, "where": None},
            "missing": [], "confidence": 0.9}


# ---- malformed model output ----

@pytest.mark.parametrize("raw", [None, [], "text", 42, {}, {"category": None},
                                 {"category": ["task"]}])
def test_garbage_model_output_is_rejected_not_crashed(raw):
    with pytest.raises(ValueError):
        validate_proposal(raw)


def test_wrong_slot_types_are_cleaned():
    result = validate_proposal({"category": "task", "intent": "do_task",
                                "slots": {"what": 123, "who": ["Ravi"], "when": {"a": 1}},
                                "confidence": "high"})
    assert result["slots"] == {"who": None, "what": None, "when": None, "where": None}
    assert result["confidence"] == 0.5
    assert result["missing"] == ["what"]


def test_adapter_falls_back_when_model_keeps_returning_bad_json(monkeypatch):
    import core.llm_adapter as adapter
    monkeypatch.setattr(adapter.ollama, "chat",
                        lambda **kw: {"message": {"content": "this is not json"}})
    result = OllamaLLMAdapter().analyze("anything")
    assert result["confidence"] == 0.0 and result["intent"] == "other"


# ---- unusual text ----

@pytest.mark.parametrize("text", ["!!!", "'; DROP TABLE notes; --", "नमस्ते रवि", "a" * 4000])
def test_odd_text_is_stored_safely(text):
    nid = make_note(text)
    r = stabilize(nid, proposal(who=None), V1)
    assert r["stability"] == "pending"
    with database.get_connection() as conn:
        assert conn.execute("SELECT text FROM notes WHERE id = ?", (nid,)).fetchone()[0] == text
        assert conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 1


# ---- repeated and out-of-order input ----

def test_same_note_sent_twice_does_not_duplicate_knowledge():
    for _ in range(2):
        r = stabilize(make_note(), proposal(when="Friday"), V1)
        fuse_fragment(r["fragment_id"])
    with database.get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM knowledge WHERE status='active'").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM knowledge_sources").fetchone()[0] == 2


def test_update_chain_keeps_exactly_one_active_version():
    first = stabilize(make_note("Call Ravi on Friday"), proposal(when="Friday"), V1)
    kid = fuse_fragment(first["fragment_id"])["knowledge_id"]
    for day in ("Monday", "Tuesday", "Wednesday"):
        frag = stabilize(make_note(f"Actually, move it to {day}"),
                         proposal(when=day), V_SIMILAR)["fragment_id"]
        res = resolve_conflict(frag, kid)
        assert res["label"] == "UPDATE"
        kid = res["knowledge_id"]
    with database.get_connection() as conn:
        statuses = [r[0] for r in conn.execute("SELECT status FROM knowledge ORDER BY id")]
    assert statuses == ["superseded", "superseded", "superseded", "active"]


def test_update_against_a_superseded_item_is_refused():
    first = stabilize(make_note("Call Ravi on Friday"), proposal(when="Friday"), V1)
    old_id = fuse_fragment(first["fragment_id"])["knowledge_id"]
    f1 = stabilize(make_note("Actually, move it to Monday"), proposal(when="Monday"), V_SIMILAR)
    resolve_conflict(f1["fragment_id"], old_id)
    f2 = stabilize(make_note("Actually, move it to Tuesday"), proposal(when="Tuesday"), V_SIMILAR)
    assert resolve_conflict(f2["fragment_id"], old_id)["label"] == "SKIPPED"


# ---- transaction safety ----

def test_failed_transaction_leaves_no_partial_rows():
    with pytest.raises(Exception):
        with database.get_connection() as conn:
            conn.execute("INSERT INTO notes (text) VALUES ('half done')")
            conn.execute("INSERT INTO fragments (note_id, stability) VALUES (9999, 'pending')")
    with database.get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 0