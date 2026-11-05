import json

import numpy as np
import pytest

from core import database
from core.embeddings import to_blob
from core.fusion import fuse_fragment, resolve_fuse_review

V1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
V_SIMILAR = np.array([0.9, 0.1, 0.0], dtype=np.float32)   # similarity ~0.99 with V1
V_MID = np.array([0.6, 0.8, 0.0], dtype=np.float32)       # similarity 0.6 with V1
V_FAR = np.array([0.0, 1.0, 0.0], dtype=np.float32)       # similarity 0.0 with V1


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    database.init_db()


def add_fragment(vec=V1, intent="call_person", category="reminder",
                 who="Ravi", when=None, what=None, stability="stable"):
    slots = {"who": who, "what": what, "when": when, "where": None}
    with database.get_connection() as conn:
        nid = conn.execute("INSERT INTO notes (text) VALUES ('x')").lastrowid
        return conn.execute(
            "INSERT INTO fragments (note_id, category, intent, summary, slots_json, "
            "embedding, stability, confidence) VALUES (?,?,?,?,?,?,?,?)",
            (nid, category, intent, "summary", json.dumps(slots), to_blob(vec), stability, 0.9),
        ).lastrowid


def count(table):
    with database.get_connection() as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_first_fragment_creates_knowledge():
    r = fuse_fragment(add_fragment(when="Friday"))
    assert r["decision"] == "new"
    assert count("knowledge") == 1
    assert count("knowledge_sources") == 1


def test_similar_fragment_is_fused_and_fills_gaps():
    fuse_fragment(add_fragment(when="Friday"))
    r = fuse_fragment(add_fragment(vec=V_SIMILAR, when="Friday", what="demo"))
    assert r["decision"] == "fused"
    assert count("knowledge") == 1
    assert count("knowledge_sources") == 2
    with database.get_connection() as conn:
        content = json.loads(conn.execute("SELECT content_json FROM knowledge").fetchone()[0])
    assert content["slots"]["what"] == "demo"


def test_unrelated_fragment_stays_separate():
    fuse_fragment(add_fragment(when="Friday"))
    r = fuse_fragment(add_fragment(vec=V_FAR, when="Friday"))
    assert r["decision"] == "new"
    assert count("knowledge") == 2


def test_different_intent_is_never_fused():
    fuse_fragment(add_fragment(when="Friday"))
    r = fuse_fragment(add_fragment(intent="buy_item", category="task", who=None, what="milk"))
    assert r["decision"] == "new"
    assert count("knowledge") == 2


def test_middle_zone_asks_the_user():
    fuse_fragment(add_fragment(when="Friday"))
    r = fuse_fragment(add_fragment(vec=V_MID, when="Friday"))
    assert r["decision"] == "ask"
    assert count("review_queue") == 1
    assert count("knowledge_sources") == 1      # not attached yet


def test_conflicting_slot_is_not_fused():
    fuse_fragment(add_fragment(when="Friday"))
    r = fuse_fragment(add_fragment(vec=V_SIMILAR, when="Monday"))
    assert r["decision"] == "conflict"
    assert count("knowledge_sources") == 1


def test_pending_fragment_is_skipped():
    r = fuse_fragment(add_fragment(stability="pending"))
    assert r["decision"] == "skipped"
    assert count("knowledge") == 0


def test_same_fragment_is_not_fused_twice():
    fid = add_fragment(when="Friday")
    fuse_fragment(fid)
    assert fuse_fragment(fid)["decision"] == "skipped"
    assert count("knowledge_sources") == 1


def test_accepting_review_fuses():
    fuse_fragment(add_fragment(when="Friday"))
    fuse_fragment(add_fragment(vec=V_MID, when="Friday"))
    assert resolve_fuse_review(1, accept=True)["decision"] == "fused"
    assert count("knowledge") == 1
    assert count("knowledge_sources") == 2


def test_rejecting_review_creates_separate_item():
    fuse_fragment(add_fragment(when="Friday"))
    fuse_fragment(add_fragment(vec=V_MID, when="Friday"))
    assert resolve_fuse_review(1, accept=False)["decision"] == "new"
    assert count("knowledge") == 2


def test_more_detailed_value_is_a_refinement_not_a_conflict():
    fuse_fragment(add_fragment(when="Friday"))
    r = fuse_fragment(add_fragment(vec=V_SIMILAR, when="Friday at 5"))
    assert r["decision"] == "fused"
    with database.get_connection() as conn:
        content = json.loads(conn.execute("SELECT content_json FROM knowledge").fetchone()[0])
    assert content["slots"]["when"] == "Friday at 5"


def test_related_intents_are_compared():
    fuse_fragment(add_fragment(when="Friday"))
    r = fuse_fragment(add_fragment(vec=V_SIMILAR, intent="schedule_meeting", when="Monday"))
    assert r["decision"] == "conflict"

def test_change_wording_on_same_person_is_a_conflict_below_high_similarity():
    fuse_fragment(add_fragment(when="Friday"))
    fid = add_fragment(vec=V_MID, when="Monday")          # similarity 0.6, between LOW and HIGH
    with database.get_connection() as conn:
        conn.execute("UPDATE notes SET text = 'Actually, move the call to Monday.' "
                     "WHERE id = (SELECT note_id FROM fragments WHERE id = ?)", (fid,))
    assert fuse_fragment(fid)["decision"] == "conflict"


def test_without_change_wording_the_middle_zone_still_asks():
    fuse_fragment(add_fragment(when="Friday"))
    assert fuse_fragment(add_fragment(vec=V_MID, when="Monday"))["decision"] == "ask"