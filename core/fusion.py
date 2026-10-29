import json
import re

from core.database import get_connection
from core.embeddings import cosine, from_blob, to_blob
from core.intents import SLOT_NAMES

HIGH = 0.75   # at or above: same knowledge (fuse automatically)
LOW = 0.50    # below: unrelated (new item); in between: ask the user


def _words(value) -> set:
    return set(re.findall(r"\w+", str(value).lower()))


def _compatible(a, b) -> bool:
    """Equal, or one is a more detailed version of the other."""
    wa, wb = _words(a), _words(b)
    return bool(wa) and bool(wb) and (wa <= wb or wb <= wa)


def _conflicts(old: dict, new: dict) -> bool:
    return any(old.get(k) and new.get(k) and not _compatible(old[k], new[k])
               for k in SLOT_NAMES)


def _fragment(conn, fragment_id):
    return conn.execute("SELECT * FROM fragments WHERE id = ?", (fragment_id,)).fetchone()


def _create(conn, frag, vec, slots) -> int:
    content = {"intent": frag["intent"], "slots": slots, "summaries": [frag["summary"]]}
    kid = conn.execute(
        "INSERT INTO knowledge (category, title, content_json, embedding) VALUES (?,?,?,?)",
        (frag["category"], frag["summary"] or frag["intent"], json.dumps(content), to_blob(vec)),
    ).lastrowid
    conn.execute("INSERT INTO knowledge_sources (knowledge_id, fragment_id) VALUES (?,?)",
                 (kid, frag["id"]))
    return kid


def _attach(conn, kid, frag, vec, slots) -> None:
    row = conn.execute("SELECT content_json, embedding FROM knowledge WHERE id = ?", (kid,)).fetchone()
    content = json.loads(row["content_json"])
    n = conn.execute("SELECT COUNT(*) FROM knowledge_sources WHERE knowledge_id = ?",
                     (kid,)).fetchone()[0]

    merged = dict(content["slots"])
    for k in SLOT_NAMES:
        new_val = slots.get(k)
        if not new_val:
            continue
        if not merged.get(k) or len(_words(new_val)) > len(_words(merged[k])):
            merged[k] = new_val               # fill a gap, or keep the more detailed value
    content["slots"] = merged
    if frag["summary"] and frag["summary"] not in content["summaries"]:
        content["summaries"].append(frag["summary"])

    new_vec = (from_blob(row["embedding"]) * n + vec) / (n + 1)   # running average
    conn.execute(
        "UPDATE knowledge SET content_json = ?, embedding = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ?", (json.dumps(content), to_blob(new_vec), kid))
    conn.execute("INSERT INTO knowledge_sources (knowledge_id, fragment_id) VALUES (?,?)",
                 (kid, frag["id"]))


def fuse_fragment(fragment_id: int) -> dict:
    """Place one stable fragment into the knowledge base."""
    with get_connection() as conn:           # one transaction
        frag = _fragment(conn, fragment_id)
        if frag is None or frag["stability"] != "stable":
            return {"decision": "skipped", "reason": "fragment missing or not stable"}
        if conn.execute("SELECT 1 FROM knowledge_sources WHERE fragment_id = ?",
                        (fragment_id,)).fetchone():
            return {"decision": "skipped", "reason": "already in knowledge"}

        vec = from_blob(frag["embedding"])
        slots = json.loads(frag["slots_json"] or "{}")

        best = None
        for row in conn.execute("SELECT id, category, content_json, embedding FROM knowledge "
                                "WHERE status = 'active' AND embedding IS NOT NULL").fetchall():
            content = json.loads(row["content_json"])
            if row["category"] != frag["category"] or content.get("intent") != frag["intent"]:
                continue                      # similarity alone doesn't prove same topic
            score = cosine(vec, from_blob(row["embedding"]))
            if best is None or score > best[0]:
                best = (score, row["id"], content)

        if best is None or best[0] < LOW:
            kid = _create(conn, frag, vec, slots)
            return {"decision": "new", "knowledge_id": kid,
                    "similarity": round(best[0], 3) if best else None}

        score, kid, content = best
        sim = round(score, 3)

        if score >= HIGH:
            if _conflicts(content["slots"], slots):
                return {"decision": "conflict", "knowledge_id": kid, "similarity": sim,
                        "reason": "slot values differ (lifecycle manager decides, Day 7)"}
            _attach(conn, kid, frag, vec, slots)
            return {"decision": "fused", "knowledge_id": kid, "similarity": sim}

        conn.execute(
            "INSERT INTO review_queue (type, payload_json) VALUES ('fuse', ?)",
            (json.dumps({"fragment_id": fragment_id, "knowledge_id": kid, "similarity": sim}),))
        return {"decision": "ask", "knowledge_id": kid, "similarity": sim}


def resolve_fuse_review(review_id: int, accept: bool) -> dict:
    """User answers a 'should these be merged?' question."""
    with get_connection() as conn:
        item = conn.execute("SELECT * FROM review_queue WHERE id = ? AND type = 'fuse' "
                            "AND status = 'pending'", (review_id,)).fetchone()
        if item is None:
            raise ValueError("No pending fuse review with that id.")
        p = json.loads(item["payload_json"])
        frag = _fragment(conn, p["fragment_id"])
        vec = from_blob(frag["embedding"])
        slots = json.loads(frag["slots_json"] or "{}")

        if accept:
            _attach(conn, p["knowledge_id"], frag, vec, slots)
            kid, decision = p["knowledge_id"], "fused"
        else:
            kid, decision = _create(conn, frag, vec, slots), "new"

        conn.execute("UPDATE review_queue SET status = ? WHERE id = ?",
                     ("accepted" if accept else "rejected", review_id))
        return {"decision": decision, "knowledge_id": kid}