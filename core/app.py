from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.database import get_connection, init_db

init_db()
app = FastAPI(title="Voice PKMS")


class TextNote(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/notes/text")
def create_text_note(note: TextNote):
    text = note.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Note text is empty.")
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO notes (text, source) VALUES (?, 'text')", (text,)
        )
        return {"id": cur.lastrowid, "text": text, "status": "new"}


@app.get("/notes")
def list_notes(limit: int = 20):
    limit = max(1, min(limit, 100))
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, text, source, created_at, status "
            "FROM notes ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


@app.get("/notes/{note_id}")
def get_note(note_id: int):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, text, source, created_at, status FROM notes WHERE id = ?",
            (note_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Note not found.")
    return dict(row)