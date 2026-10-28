import json

import numpy as np
import pytest

from core import database
from core.intents import REQUIRED_SLOTS
from core.stabilization import (confirm_fragment, find_stale_pending, stabilize)

V1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
V_SIMILAR = np.array([0.9, 0.1, 0.0], dtype=np.float32)
V_FAR = np.array([0.0, 1.0, 0.0], dtype=np.float32)


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    database.init_db()


def make_note():
    with database.get_connection() as conn:
        return conn.execute("INSERT INTO notes (text) VALUES ('x')").lastrowid


def proposal(intent="call_person", who=None, what=None, when=None, confidence=0.9):
    slots = {"who": who, "what": what, "when": when, "where": None}
    missing = [s for s in REQUIRED_SLOTS[intent] if slots[s] is None]
    return {"category": "reminder", "intent": intent, "summary": "s",
            "slots": slots, "missing": missing, "confidence": confidence}


def get_fragment(fid):
    with database.get_connection() as conn:
        return dict(conn.execute("SELECT * FROM fragments WHERE id = ?", (fid,)).fetchone())


def test_complete_fragment_is_stable():
    r = stabilize(make_note(), proposal(who="Ravi", when="Friday"), V1)
    assert r["stability"] == "stable"


def test_incomplete_fragment_is_pending():
    r = stabilize(make_note(), proposal(who="Ravi"), V1)
    assert r["stability"] == "pending"
    assert "when" in r["reason"]


def test_failed_analysis_is_pending():
    r = stabilize(make_note(), proposal(who="Ravi", when="Friday", confidence=0.0), V1)
    assert r["stability"] == "pending"
    assert r["reason"] == "analysis_failed"


def test_later_fragment_fills_pending_one():
    first = stabilize(make_note(), proposal(who="Ravi"), V1)
    second = stabilize(make_note(), proposal(who="Ravi", when="Friday at 5"), V_SIMILAR)

    assert second["stability"] == "absorbed"
    assert second["merged_into"] == first["fragment_id"]

    target = get_fragment(first["fragment_id"])
    assert target["stability"] == "stable"
    assert json.loads(target["slots_json"])["when"] == "Friday at 5"


def test_dissimilar_fragment_is_not_merged():
    stabilize(make_note(), proposal(who="Ravi"), V1)
    r = stabilize(make_note(), proposal(who="Ravi", when="Friday"), V_FAR)
    assert r["merged_into"] is None
    assert r["stability"] == "stable"


def test_conflicting_slot_is_not_merged():
    stabilize(make_note(), proposal(who="Ravi"), V1)
    r = stabilize(make_note(), proposal(who="Sam", when="Friday"), V_SIMILAR)
    assert r["merged_into"] is None


def test_confirm_makes_pending_stable():
    r = stabilize(make_note(), proposal(who="Ravi"), V1)
    assert confirm_fragment(r["fragment_id"]) is True
    assert get_fragment(r["fragment_id"])["stability"] == "stable"


def test_old_pending_fragments_are_found():
    r = stabilize(make_note(), proposal(who="Ravi"), V1)
    with database.get_connection() as conn:
        conn.execute("UPDATE fragments SET created_at = '2000-01-01 00:00:00' WHERE id = ?",
                     (r["fragment_id"],))
    assert [f["id"] for f in find_stale_pending(24)] == [r["fragment_id"]]