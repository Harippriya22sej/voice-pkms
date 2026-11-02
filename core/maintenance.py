import json

from core.database import get_connection
from core.stabilization import find_stale_pending, missing_slots

STALE_HOURS = 24


def queue_stale_pending(hours: int = STALE_HOURS) -> int:
    """Turn long-waiting pending fragments into review questions. Returns how many were added."""
    stale = find_stale_pending(hours)
    added = 0
    with get_connection() as conn:
        for f in stale:
            already = conn.execute(
                "SELECT 1 FROM review_queue WHERE type = 'stabilize' AND status = 'pending' "
                "AND json_extract(payload_json, '$.fragment_id') = ?", (f["id"],)).fetchone()
            if already:
                continue                       # never queue the same question twice
            slots = json.loads(f["slots_json"] or "{}")
            payload = {"fragment_id": f["id"], "missing": missing_slots(f["intent"], slots)}
            conn.execute("INSERT INTO review_queue (type, payload_json) VALUES ('stabilize', ?)",
                         (json.dumps(payload),))
            added += 1
    return added


def resolve_stale_review(review_id: int, action: str) -> dict:
    """'confirm' makes the fragment stable as it is; 'keep_waiting' restarts its clock."""
    if action not in ("confirm", "keep_waiting"):
        raise ValueError("action must be 'confirm' or 'keep_waiting'")
    with get_connection() as conn:
        item = conn.execute("SELECT * FROM review_queue WHERE id = ? AND type = 'stabilize' "
                            "AND status = 'pending'", (review_id,)).fetchone()
        if item is None:
            raise ValueError("No pending stabilization review with that id.")
        fragment_id = json.loads(item["payload_json"])["fragment_id"]

        if action == "confirm":
            cur = conn.execute("UPDATE fragments SET stability = 'stable' "
                               "WHERE id = ? AND stability = 'pending'", (fragment_id,))
            if cur.rowcount == 0:
                raise ValueError("That fragment is no longer pending.")
        else:
            conn.execute("UPDATE fragments SET created_at = CURRENT_TIMESTAMP "
                         "WHERE id = ? AND stability = 'pending'", (fragment_id,))

        conn.execute("UPDATE review_queue SET status = ? WHERE id = ?",
                     ("accepted" if action == "confirm" else "rejected", review_id))
    return {"fragment_id": fragment_id, "action": action}