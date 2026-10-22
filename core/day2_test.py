import sqlite3
from core.database import get_connection, init_db

init_db()

with get_connection() as conn:
    # Insert a note. The ? placeholders keep user text out of the SQL itself.
    cur = conn.execute(
        "INSERT INTO notes (text, source) VALUES (?, ?)",
        ("Remind me to call Ravi about the demo.", "text"),
    )
    note_id = cur.lastrowid

    # Insert a fragment linked to that note.
    conn.execute(
        "INSERT INTO fragments (note_id, category, intent, stability) VALUES (?, ?, ?, ?)",
        (note_id, "task", "call_person", "pending"),
    )

    # Read it back.
    rows = conn.execute(
        "SELECT n.id, n.text, f.intent, f.stability "
        "FROM notes n JOIN fragments f ON f.note_id = n.id"
    ).fetchall()
    for row in rows:
        print(dict(row))

    # Update: mark the fragment as stable.
    conn.execute("UPDATE fragments SET stability = ? WHERE note_id = ?", ("stable", note_id))

# Foreign-key check: this must FAIL because note 9999 doesn't exist.
try:
    with get_connection() as conn:
        conn.execute("INSERT INTO fragments (note_id) VALUES (?)", (9999,))
except sqlite3.IntegrityError as e:
    print("Good, the database rejected a bad link:", e)