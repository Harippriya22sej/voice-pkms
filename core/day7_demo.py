from core.database import get_connection
from core.pipeline import process_text_note

notes = [
    "Call Ravi on Friday at 5.",                 # new knowledge
    "Actually, move the Ravi call to Monday.",   # explicit change -> UPDATE, version 2
    "Call Ravi on Tuesday.",                     # differs, no change wording -> review
]

for text in notes:
    r = process_text_note(text)
    print("\n", text)
    print("  fusion   :", r["fusion"])
    print("  lifecycle:", r["lifecycle"])

with get_connection() as conn:
    print("\nKNOWLEDGE")
    for row in conn.execute("SELECT id, version, status, supersedes_id, content_json FROM knowledge"):
        print(" ", dict(row))
    print("REVIEW QUEUE:", [dict(r) for r in conn.execute(
        "SELECT id, type, status, payload_json FROM review_queue")])