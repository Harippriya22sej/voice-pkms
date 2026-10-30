import json

import numpy as np
import pytest

from core import database
from core.embeddings import to_blob
from core.fusion import fuse_fragment
from core.lifecycle import (decide, differing_slots, has_update_cue, history,
                            resolve_conflict, resolve_contradiction_review)

V1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
V_SIMILAR = np.array([0.9, 0.1, 0.0], dtype=np.float32)


class FakeJudge:
    def __init__(self, label):
        self.label = label

    def judge(self, **kwargs):
        return self.label


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    database.init_db()


def add_fragment(text, vec=V1, when="Friday", who="Ravi"):
    slots = {"who": who, "what": None, "when": when, "where": None}
    with database.get_connection() as conn:
        nid = conn.execute("INSERT INTO notes (text) VALUES (?)", (text,)).lastrowid
        return conn.execute(
            "INSERT INTO fragments (note_id, category, intent, summary, slots_json, "
            "embedding, stability, confidence) VALUES (?,?,?,?,?,?,?,?)",
            (nid, "reminder", "call_person", text, json.dumps(slots),
             to_blob(vec), "stable", 0.9),
        ).lastrowid


def base_knowledge():
    return fuse_fragment(add_fragment("Call Ravi on Friday"))["knowledge_id"]


def rows(sql):
    with database.get_connection() as conn:
        return [dict(r) for r in conn.execute(sql).fetchall()]


def test_differing_slots_ignores_refinements():
    old = {"who": "Ravi", "when": "Friday"}
    assert differing_slots(old, {"who": "Ravi", "when": "Friday at 5"}) == {}
    assert "when" in differing_slots(old, {"who": "Ravi", "when": "Monday"})


def test_update_cues():
    assert has_update_cue("Actually, move the call to Monday.")
    assert not has_update_cue("Call Ravi on Monday.")


def test_no_difference_is_support():
    assert decide({"when": "Friday"}, {"when": "Friday at 5"}, "call Ravi")["label"] == "SUPPORT"


def test_explicit_change_creates_new_version():
    kid = base_knowledge()
    frag = add_fragment("Actually, move the call to Monday.", vec=V_SIMILAR, when="Monday")
    r = resolve_conflict(frag, kid)

    assert r["label"] == "UPDATE" and r["version"] == 2
    items = rows("SELECT id, status, version, supersedes_id FROM knowledge ORDER BY id")
    assert items[0]["status"] == "superseded"           # old version is kept
    assert items[1]["status"] == "active" and items[1]["supersedes_id"] == kid
    content = json.loads(rows("SELECT content_json FROM knowledge WHERE version = 2")[0]["content_json"])
    assert content["slots"]["when"] == "Monday"
    assert content["changes"]["when"] == {"from": "Friday", "to": "Monday"}


def test_difference_without_cue_goes_to_review_and_changes_nothing():
    kid = base_knowledge()
    frag = add_fragment("Call Ravi on Monday.", vec=V_SIMILAR, when="Monday")
    r = resolve_conflict(frag, kid)

    assert r["label"] == "CONTRADICT"
    queue = rows("SELECT type, status FROM review_queue")
    assert queue == [{"type": "contradict", "status": "pending"}]
    assert rows("SELECT status FROM knowledge") == [{"status": "active"}]


def test_judge_cannot_trigger_an_update():
    kid = base_knowledge()
    frag = add_fragment("Call Ravi on Monday.", vec=V_SIMILAR, when="Monday")
    r = resolve_conflict(frag, kid, judge=FakeJudge("UPDATE"))
    assert r["label"] == "CONTRADICT"


def test_judge_can_mark_unrelated():
    kid = base_knowledge()
    frag = add_fragment("Call Ravi on Monday.", vec=V_SIMILAR, when="Monday")
    r = resolve_conflict(frag, kid, judge=FakeJudge("UNRELATED"))
    assert r["label"] == "UNRELATED"
    assert [k["status"] for k in rows("SELECT status FROM knowledge")] == ["active", "active"]


def test_support_attaches_to_existing_item():
    kid = base_knowledge()
    frag = add_fragment("Call Ravi Friday at 5", vec=V_SIMILAR, when="Friday at 5")
    assert resolve_conflict(frag, kid)["label"] == "SUPPORT"
    assert len(rows("SELECT * FROM knowledge")) == 1
    assert len(rows("SELECT * FROM knowledge_sources")) == 2


def test_same_fragment_is_not_resolved_twice():
    kid = base_knowledge()
    frag = add_fragment("Actually, move the call to Monday.", vec=V_SIMILAR, when="Monday")
    resolve_conflict(frag, kid)
    assert resolve_conflict(frag, kid)["label"] == "SKIPPED"


def test_review_accept_update():
    kid = base_knowledge()
    resolve_conflict(add_fragment("Call Ravi on Monday.", vec=V_SIMILAR, when="Monday"), kid)
    resolve_contradiction_review(1, "update")
    assert rows("SELECT status FROM knowledge WHERE id = 1") == [{"status": "superseded"}]
    assert rows("SELECT status FROM review_queue") == [{"status": "accepted"}]


def test_review_keep_old_changes_nothing():
    kid = base_knowledge()
    resolve_conflict(add_fragment("Call Ravi on Monday.", vec=V_SIMILAR, when="Monday"), kid)
    resolve_contradiction_review(1, "keep_old")
    assert len(rows("SELECT * FROM knowledge")) == 1
    assert rows("SELECT status FROM review_queue") == [{"status": "rejected"}]


def test_review_separate_creates_second_item():
    kid = base_knowledge()
    resolve_conflict(add_fragment("Call Ravi on Monday.", vec=V_SIMILAR, when="Monday"), kid)
    resolve_contradiction_review(1, "separate")
    assert [k["status"] for k in rows("SELECT status FROM knowledge")] == ["active", "active"]


def test_history_lists_versions_oldest_first():
    kid = base_knowledge()
    resolve_conflict(add_fragment("Actually, move it to Monday.", vec=V_SIMILAR, when="Monday"), kid)
    assert [v["version"] for v in history(kid)] == [1, 2]
    assert [v["version"] for v in history(2)] == [1, 2]