from core.database import get_connection
from core.pipeline import process_text_note

notes = [
    "Call Ravi on Friday at 5.",                              # new knowledge
    "Buy vegetables from the market tomorrow.",               # separate knowledge
    "Remember to call Ravi on Friday at 5 about the demo.",   # should fuse with #1
    "Call Ravi on Monday.",                                   # similar but conflicting
]

for text in notes:
    result = process_text_note(text)
    print("\n", text)
    print("  stability:", result["stability"], "| fusion:", result["fusion"])

with get_connection() as conn:
    print("\nKNOWLEDGE")
    for row in conn.execute("SELECT id, title, status, content_json FROM knowledge"):
        print(" ", dict(row))
    print("SOURCES:", [tuple(r) for r in conn.execute("SELECT * FROM knowledge_sources")])
    print("REVIEW QUEUE:", [dict(r) for r in conn.execute("SELECT * FROM review_queue")])