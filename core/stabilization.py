import json

from core.database import get_connection
from core.embeddings import cosine, from_blob, to_blob
from core.intents import REQUIRED_SLOTS, SLOT_NAMES

FILL_THRESHOLD = 0.45   # starting guess; tune it with your own notes
MIN_CONFIDENCE = 0.1    # below this the model call failed (adapter fallback)


def missing_slots(intent: str, slots: dict) -> list:
    return [s for s in REQUIRED_SLOTS.get(intent, []) if not slots.get(s)]


def judge(proposal: dict) -> tuple[str, str]:
    """Decide stable / pending for one fragment, with a reason."""
    if proposal["confidence"] < MIN_CONFIDENCE:
        return "pending", "analysis_failed"
    missing = missing_slots(proposal["intent"], proposal["slots"])
    if missing:
        return "pending", "missing: " + ", ".join(missing)
    return "stable", "complete"


def _same(a, b) -> bool:
    return str(a).strip().lower() == str(b).strip().lower()


def _find_fill_target(conn, proposal: dict, vec):
    """Find the pending fragment this new fragment can complete, if any."""
    new_slots = proposal["slots"]
    best = None
    rows = conn.execute(
        "SELECT id, intent, slots_json, embedding FROM fragments "
        "WHERE stability = 'pending' AND embedding IS NOT NULL"
    ).fetchall()
    for row in rows:
        # Compatible intent: same, or the new fragment is a vague "other".
        if row["intent"] != proposal["intent"] and proposal["intent"] != "other":
            continue
        old_slots = json.loads(row["slots_json"] or "{}")

        # A direct disagreement is not slot-filling (Day 7 handles conflicts).
        if any(old_slots.get(k) and new_slots.get(k) and not _same(old_slots[k], new_slots[k])
               for k in SLOT_NAMES):
            continue

        fills = [k for k in SLOT_NAMES if not old_slots.get(k) and new_slots.get(k)]
        if not fills:
            continue

        score = cosine(vec, from_blob(row["embedding"]))
        if score < FILL_THRESHOLD:
            continue
        if best is None or score > best[0]:
            best = (score, row, old_slots, fills)
    return best


def _insert_fragment(conn, note_id, proposal, vec, stability, merged_into=None) -> int:
    cur = conn.execute(
        "INSERT INTO fragments (note_id, category, intent, summary, slots_json, "
        "embedding, stability, merged_into, confidence) VALUES (?,?,?,?,?,?,?,?,?)",
        (note_id, proposal["category"], proposal["intent"], proposal["summary"],
         json.dumps(proposal["slots"]), to_blob(vec), stability, merged_into,
         proposal["confidence"]),
    )
    return cur.lastrowid


def stabilize(note_id: int, proposal: dict, vec) -> dict:
    """Store one analyzed fragment and decide its stability."""
    with get_connection() as conn:   # one transaction: all changes or none
        target = _find_fill_target(conn, proposal, vec)

        if target:
            score, row, old_slots, fills = target
            merged = dict(old_slots)
            for k in fills:
                merged[k] = proposal["slots"][k]
            target_stability = "pending" if missing_slots(row["intent"], merged) else "stable"
            new_vec = (from_blob(row["embedding"]) + vec) / 2   # blend both meanings

            conn.execute(
                "UPDATE fragments SET slots_json = ?, embedding = ?, stability = ? WHERE id = ?",
                (json.dumps(merged), to_blob(new_vec), target_stability, row["id"]),
            )
            fid = _insert_fragment(conn, note_id, proposal, vec, "absorbed",
                                   merged_into=row["id"])
            return {"fragment_id": fid, "stability": "absorbed",
                    "merged_into": row["id"], "target_stability": target_stability,
                    "similarity": round(score, 3), "reason": "filled: " + ", ".join(fills)}

        stability, reason = judge(proposal)
        fid = _insert_fragment(conn, note_id, proposal, vec, stability)
        return {"fragment_id": fid, "stability": stability, "merged_into": None,
                "target_stability": None, "similarity": None, "reason": reason}


def confirm_fragment(fragment_id: int) -> bool:
    """User manually accepts a pending fragment as stable."""
    with get_connection() as conn:
        cur = conn.execute(
            "UPDATE fragments SET stability = 'stable' "
            "WHERE id = ? AND stability = 'pending'", (fragment_id,))
        return cur.rowcount > 0


def find_stale_pending(hours: int = 24) -> list:
    """Pending fragments older than N hours (Day 10 sends these to the review queue)."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, note_id, intent, slots_json, created_at FROM fragments "
            "WHERE stability = 'pending' AND created_at < datetime('now', ?)",
            (f"-{int(hours)} hours",),
        ).fetchall()
    return [dict(r) for r in rows]