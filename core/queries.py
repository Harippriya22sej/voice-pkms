import json

from core.database import get_connection
from core.lifecycle import history as _history
from core.stabilization import missing_slots


def _knowledge_dict(row) -> dict:
    d = dict(row)
    d.pop("embedding", None)                       # binary data, not for the API
    d["content"] = json.loads(d.pop("content_json") or "{}")
    return d


def list_pending() -> list:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT f.id, f.note_id, f.category, f.intent, f.summary, f.slots_json, "
            "f.created_at, n.text FROM fragments f JOIN notes n ON n.id = f.note_id "
            "WHERE f.stability = 'pending' ORDER BY f.id DESC").fetchall()
    out = []
    for r in rows:
        slots = json.loads(r["slots_json"] or "{}")
        out.append({"id": r["id"], "note_id": r["note_id"], "text": r["text"],
                    "category": r["category"], "intent": r["intent"],
                    "summary": r["summary"], "slots": slots,
                    "missing": missing_slots(r["intent"], slots),
                    "created_at": r["created_at"]})
    return out


def list_knowledge(status: str = "active", limit: int = 50, category: str | None = None) -> list:
    limit = max(1, min(limit, 200))
    sql = "SELECT * FROM knowledge"
    clauses, params = [], []
    if status != "all":
        clauses.append("status = ?")
        params.append(status)
    if category:
        clauses.append("category = ?")
        params.append(category)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_knowledge_dict(r) for r in rows]


def get_knowledge(knowledge_id: int):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM knowledge WHERE id = ?", (knowledge_id,)).fetchone()
        if row is None:
            return None
        sources = conn.execute(
            "SELECT f.id AS fragment_id, n.id AS note_id, n.text, n.created_at "
            "FROM knowledge_sources ks JOIN fragments f ON f.id = ks.fragment_id "
            "JOIN notes n ON n.id = f.note_id WHERE ks.knowledge_id = ? ORDER BY f.id",
            (knowledge_id,)).fetchall()
    versions = [_knowledge_dict(v) for v in _history(knowledge_id)]
    return {"item": _knowledge_dict(row), "sources": [dict(s) for s in sources],
            "history": versions}


def list_review(status: str = "pending") -> list:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM review_queue WHERE status = ? ORDER BY id",
                            (status,)).fetchall()
        out = []
        for r in rows:
            payload = json.loads(r["payload_json"] or "{}")
            context = {}
            if payload.get("fragment_id"):
                f = conn.execute(
                    "SELECT f.slots_json, n.text FROM fragments f JOIN notes n "
                    "ON n.id = f.note_id WHERE f.id = ?", (payload["fragment_id"],)).fetchone()
                if f:
                    context["new_note"] = f["text"]
                    context["new_slots"] = json.loads(f["slots_json"] or "{}")
            if payload.get("knowledge_id"):
                k = conn.execute("SELECT title, version, status, content_json FROM knowledge "
                                 "WHERE id = ?", (payload["knowledge_id"],)).fetchone()
                if k:
                    context["existing"] = {
                        "title": k["title"], "version": k["version"], "status": k["status"],
                        "slots": json.loads(k["content_json"] or "{}").get("slots", {})}
            out.append({"id": r["id"], "type": r["type"], "status": r["status"],
                        "created_at": r["created_at"], "payload": payload,
                        "context": context})
    return out