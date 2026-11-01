import json

import numpy as np
import pytest

from core import database, retrieval
from core.embeddings import to_blob


def fake_embed(text):
    lower = text.lower()
    if any(w in lower for w in ("ravi", "phone", "call")):
        return np.array([1, 0, 0], dtype=np.float32)
    if "vegetables" in lower:
        return np.array([0, 1, 0], dtype=np.float32)
    return np.array([0, 0, 1], dtype=np.float32)


@pytest.fixture(autouse=True)
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(retrieval, "embed", fake_embed)
    database.init_db()


def add_item(title, status="active", version=1):
    content = {"intent": "x", "slots": {"who": None, "what": None, "when": None, "where": None},
               "summaries": [title]}
    with database.get_connection() as conn:
        conn.execute(
            "INSERT INTO knowledge (category, title, content_json, embedding, status, version) "
            "VALUES (?,?,?,?,?,?)",
            ("reminder", title, json.dumps(content), to_blob(fake_embed(title)), status, version))


def test_finds_item_by_meaning_without_shared_words():
    add_item("Call Ravi on Friday")
    add_item("Buy vegetables tomorrow")
    results = retrieval.search("phone someone")
    assert [r["title"] for r in results] == ["Call Ravi on Friday"]
    assert results[0]["keyword"] == 0.0 and results[0]["semantic"] > 0.9


def test_unrelated_query_returns_nothing():
    add_item("Call Ravi on Friday")
    assert retrieval.search("weather forecast") == []


def test_superseded_items_hidden_unless_requested():
    add_item("Call Ravi on Friday", status="superseded")
    add_item("Call Ravi on Monday", version=2)
    assert [r["title"] for r in retrieval.search("Ravi")] == ["Call Ravi on Monday"]
    assert len(retrieval.search("Ravi", include_superseded=True)) == 2


def test_empty_query_is_rejected():
    with pytest.raises(ValueError):
        retrieval.search("   ")


def test_limit_is_respected():
    for i in range(5):
        add_item(f"Call Ravi number {i}")
    assert len(retrieval.search("Ravi", limit=2)) == 2