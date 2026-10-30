import json

import ollama

from core.database import get_connection
from core.embeddings import from_blob, to_blob
from core.fusion import _attach, _compatible, _create, _fragment, _words
from core.intents import SLOT_NAMES

LABELS = ["SUPPORT", "UPDATE", "CONTRADICT", "UNRELATED"]

# Wording that signals "the user is changing something they said before".
UPDATE_CUES = {"actually", "changed", "change", "moved", "move", "instead",
               "reschedule", "rescheduled", "postponed", "postpone", "shifted",
               "switched", "updated", "correction"}


def differing_slots(old: dict, new: dict) -> dict:
    """Slots where both sides have a value and the values are not compatible."""
    return {k: {"from": old[k], "to": new[k]} for k in SLOT_NAMES
            if old.get(k) and new.get(k) and not _compatible(old[k], new[k])}


def has_update_cue(text: str) -> bool:
    return bool(_words(text) & UPDATE_CUES)


class OllamaLifecycleJudge:
    """Asks the local LLM how the new note relates to stored knowledge."""

    def __init__(self, model: str = "qwen2.5:3b"):
        self.model = model

    def judge(self, old_summary, old_slots, new_text, diff) -> str:
        prompt = (
            f"Stored knowledge: {old_summary} | details: {json.dumps(old_slots)}\n"
            f"New note: {new_text}\n"
            f"Details that differ: {json.dumps(diff)}\n\n"
            "How does the new note relate to the stored knowledge? Answer with one label:\n"
            "- SUPPORT: confirms it\n"
            "- UPDATE: replaces a detail with a newer one\n"
            "- CONTRADICT: conflicts with it\n"
            "- UNRELATED: is about a different thing"
        )
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                format={"type": "object",
                        "properties": {"label": {"type": "string", "enum": LABELS}},
                        "required": ["label"]},
                options={"temperature": 0},
            )
            label = json.loads(response["message"]["content"]).get("label")
            return label if label in LABELS else "CONTRADICT"
        except Exception:
            return "CONTRADICT"   # safe default: ask the user


def decide(old_slots: dict, new_slots: dict, new_text: str, judge=None,
           old_summary: str = "") -> dict:
    """Python policy. The LLM can only veto (UNRELATED), never trigger an update."""
    diff = differing_slots(old_slots, new_slots)
    if not diff:
        return {"label": "SUPPORT", "diff": diff, "reason": "no stored detail is contradicted"}
    if has_update_cue(new_text):
        return {"label": "UPDATE", "diff": diff, "reason": "explicit change wording"}
    if judge is not None:
        opinion = judge.judge(old_summary=old_summary, old_slots=old_slots,
                              new_text=new_text, diff=diff)
        if opinion == "UNRELATED":
            return {"label": "UNRELATED", "diff": diff, "reason": "judge: different subject"}
    return {"label": "CONTRADICT", "diff": diff,
            "reason": "values differ without explicit change wording"}


def _note_text(conn, frag) -> str:
    rows = conn.execute(
        "SELECT text FROM notes WHERE id = ? OR id IN "
        "(SELECT note_id FROM fragments WHERE merged_into = ?) ORDER BY id",
        (frag["note_id"], frag["id"])).fetchall()
    return " ".join(r["text"] for r in rows)


def _apply_update(conn, old, frag, vec, slots, diff) -> int:
    """Create version N+1 and supersede the old one. The old row is kept."""
    content = json.loads(old["content_json"])
    merged = dict(content["slots"])
    for k in SLOT_NAMES:
        if k in diff or (slots.get(k) and not merged.get(k)):
            merged[k] = slots[k]
    new_content = {"intent": content.get("intent"), "slots": merged,
                   "summaries": [frag["summary"]], "changes": diff}

    new_id = conn.execute(
        "INSERT INTO knowledge (category, title, content_json, embedding, version, supersedes_id) "
        "VALUES (?,?,?,?,?,?)",
        (old["category"], frag["summary"] or old["title"], json.dumps(new_content),
         to_blob(vec), old["version"] + 1, old["id"])).lastrowid
    conn.execute("UPDATE knowledge SET status = 'superseded', updated_at = CURRENT_TIMESTAMP "
                 "WHERE id = ?", (old["id"],))
    conn.execute("INSERT INTO knowledge_sources (knowledge_id, fragment_id) VALUES (?,?)",
                 (new_id, frag["id"]))
    return new_id


def resolve_conflict(fragment_id: int, knowledge_id: int, judge=None) -> dict:
    """Decide what a conflicting stable fragment means for existing knowledge."""
    with get_connection() as conn:   # one transaction
        frag = _fragment(conn, fragment_id)
        old = conn.execute("SELECT * FROM knowledge WHERE id = ?", (knowledge_id,)).fetchone()
        if frag is None or old is None or old["status"] != "active" or frag["stability"] != "stable":
            return {"label": "SKIPPED", "reason": "fragment or knowledge not eligible"}
        if conn.execute("SELECT 1 FROM knowledge_sources WHERE fragment_id = ?",
                        (fragment_id,)).fetchone():
            return {"label": "SKIPPED", "reason": "already in knowledge"}

        vec = from_blob(frag["embedding"])
        slots = json.loads(frag["slots_json"] or "{}")
        content = json.loads(old["content_json"])
        result = decide(content["slots"], slots, _note_text(conn, frag), judge, old["title"])
        label, diff, reason = result["label"], result["diff"], result["reason"]

        if label == "SUPPORT":
            _attach(conn, knowledge_id, frag, vec, slots)
            return {"label": label, "knowledge_id": knowledge_id, "reason": reason}
        if label == "UPDATE":
            new_id = _apply_update(conn, old, frag, vec, slots, diff)
            return {"label": label, "knowledge_id": new_id, "supersedes": knowledge_id,
                    "version": old["version"] + 1, "changes": diff, "reason": reason}
        if label == "UNRELATED":
            new_id = _create(conn, frag, vec, slots)
            return {"label": label, "knowledge_id": new_id, "reason": reason}

        conn.execute(
            "INSERT INTO review_queue (type, payload_json) VALUES ('contradict', ?)",
            (json.dumps({"fragment_id": fragment_id, "knowledge_id": knowledge_id,
                         "changes": diff, "reason": reason}),))
        return {"label": "CONTRADICT", "knowledge_id": knowledge_id,
                "changes": diff, "reason": reason}


def resolve_contradiction_review(review_id: int, action: str) -> dict:
    """User answers a contradiction: 'update', 'keep_old' or 'separate'."""
    if action not in ("update", "keep_old", "separate"):
        raise ValueError("action must be 'update', 'keep_old' or 'separate'")
    with get_connection() as conn:
        item = conn.execute("SELECT * FROM review_queue WHERE id = ? AND type = 'contradict' "
                            "AND status = 'pending'", (review_id,)).fetchone()
        if item is None:
            raise ValueError("No pending contradiction review with that id.")
        p = json.loads(item["payload_json"])
        frag = _fragment(conn, p["fragment_id"])
        old = conn.execute("SELECT * FROM knowledge WHERE id = ?", (p["knowledge_id"],)).fetchone()
        vec = from_blob(frag["embedding"])
        slots = json.loads(frag["slots_json"] or "{}")
        result = {"action": action}

        if action == "keep_old":
            status = "rejected"
        else:
            status = "accepted"
            if action == "update":
                if old["status"] != "active":
                    raise ValueError("That knowledge item was already superseded.")
                diff = differing_slots(json.loads(old["content_json"])["slots"], slots)
                result["knowledge_id"] = _apply_update(conn, old, frag, vec, slots, diff)
            else:
                result["knowledge_id"] = _create(conn, frag, vec, slots)

        conn.execute("UPDATE review_queue SET status = ? WHERE id = ?", (status, review_id))
        return result


def history(knowledge_id: int) -> list:
    """All versions of one knowledge item, oldest first."""
    with get_connection() as conn:
        cur = conn.execute("SELECT * FROM knowledge WHERE id = ?", (knowledge_id,)).fetchone()
        if cur is None:
            return []
        while cur["supersedes_id"] is not None:
            cur = conn.execute("SELECT * FROM knowledge WHERE id = ?",
                               (cur["supersedes_id"],)).fetchone()
        chain = [dict(cur)]
        while True:
            nxt = conn.execute("SELECT * FROM knowledge WHERE supersedes_id = ?",
                               (cur["id"],)).fetchone()
            if nxt is None:
                return chain
            chain.append(dict(nxt))
            cur = nxt