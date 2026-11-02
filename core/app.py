import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from core import queries, retrieval
from core.database import BASE_DIR, get_connection, init_db
from core.fusion import resolve_fuse_review
from core.intents import CATEGORIES
from core.lifecycle import OllamaLifecycleJudge, resolve_contradiction_review
from core.llm_adapter import OllamaLLMAdapter
from core.maintenance import STALE_HOURS, queue_stale_pending, resolve_stale_review
from core.pipeline import integrate_fragment, process_text_note
from core.speech import FasterWhisperAdapter
from core.stabilization import confirm_fragment

AUDIO_DIR = BASE_DIR / "data" / "audio"
ALLOWED_AUDIO = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac"}
MAX_AUDIO_BYTES = 25 * 1024 * 1024

# Created lazily; tests replace these with fakes.
state = {"llm": None, "judge": None, "speech": None}


def get_llm():
    if state["llm"] is None:
        state["llm"] = OllamaLLMAdapter()
    return state["llm"]


def get_judge():
    if state["judge"] is None:
        state["judge"] = OllamaLifecycleJudge()
    return state["judge"]


def get_speech():
    if state["speech"] is None:
        state["speech"] = FasterWhisperAdapter("base")
    return state["speech"]


@asynccontextmanager
async def lifespan(app):
    init_db()
    queue_stale_pending(STALE_HOURS)     # old pending notes become review questions
    yield


app = FastAPI(title="Voice PKMS", lifespan=lifespan)

PROCESSING_ERROR = ("Processing failed. Your note was saved; make sure Ollama is "
                    "running, then try again.")


class TextNote(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


class ReviewResolution(BaseModel):
    action: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    with get_connection() as conn:
        def n(sql):
            return conn.execute(sql).fetchone()[0]
        return {
            "notes": n("SELECT COUNT(*) FROM notes"),
            "pending_fragments": n("SELECT COUNT(*) FROM fragments WHERE stability = 'pending'"),
            "active_knowledge": n("SELECT COUNT(*) FROM knowledge WHERE status = 'active'"),
            "superseded_knowledge": n("SELECT COUNT(*) FROM knowledge WHERE status = 'superseded'"),
            "pending_reviews": n("SELECT COUNT(*) FROM review_queue WHERE status = 'pending'"),
        }


@app.post("/notes/text")
def create_text_note(note: TextNote):
    try:
        return process_text_note(note.text, llm=get_llm(), judge=get_judge())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=503, detail=PROCESSING_ERROR)


@app.post("/notes/voice")
def create_voice_note(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_AUDIO:
        raise HTTPException(status_code=415,
                            detail="Unsupported file type. Use: " + ", ".join(sorted(ALLOWED_AUDIO)))

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    path = AUDIO_DIR / f"{uuid.uuid4().hex}{ext}"   # never trust the uploaded file name
    size, too_large = 0, False
    with open(path, "wb") as out:
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_AUDIO_BYTES:
                too_large = True
                break
            out.write(chunk)
    if too_large or size == 0:
        path.unlink(missing_ok=True)
        if too_large:
            raise HTTPException(status_code=413, detail="Audio file is too large.")
        raise HTTPException(status_code=400, detail="Audio file is empty.")

    try:
        transcript = get_speech().transcribe(path)
    except Exception:
        raise HTTPException(status_code=503,
                            detail="Transcription failed. The audio was saved; please try again.")
    if not transcript:
        raise HTTPException(status_code=422, detail="No speech detected in the recording.")

    try:
        result = process_text_note(transcript, llm=get_llm(), judge=get_judge(),
                                   source="voice", audio_path=str(path))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=503, detail=PROCESSING_ERROR)
    return {"transcript": transcript, **result}


@app.get("/search")
def search(q: str, limit: int = 5, include_old: bool = False):
    try:
        return retrieval.search(q, limit=limit, include_superseded=include_old)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/notes")
def list_notes(limit: int = 20):
    limit = max(1, min(limit, 100))
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, text, source, created_at, status "
            "FROM notes ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


@app.get("/notes/{note_id}")
def get_note(note_id: int):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, text, source, created_at, status FROM notes WHERE id = ?",
            (note_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Note not found.")
    return dict(row)


@app.get("/pending")
def pending():
    return queries.list_pending()


@app.post("/pending/{fragment_id}/confirm")
def confirm_pending(fragment_id: int):
    if not confirm_fragment(fragment_id):
        raise HTTPException(status_code=404, detail="No pending fragment with that id.")
    try:
        fusion, lifecycle = integrate_fragment(fragment_id, judge=get_judge())
    except Exception:
        raise HTTPException(status_code=503, detail=PROCESSING_ERROR)
    return {"fragment_id": fragment_id, "fusion": fusion, "lifecycle": lifecycle}


@app.post("/maintenance/stale-check")
def stale_check(hours: int = STALE_HOURS):
    if not 0 <= hours <= 720:
        raise HTTPException(status_code=400, detail="hours must be between 0 and 720")
    return {"queued": queue_stale_pending(hours)}


@app.get("/knowledge")
def knowledge(status: str = "active", limit: int = 50, category: str | None = None):
    if status not in ("active", "superseded", "all"):
        raise HTTPException(status_code=400, detail="status must be active, superseded or all")
    if category is not None and category not in CATEGORIES:
        raise HTTPException(status_code=400,
                            detail="category must be one of: " + ", ".join(CATEGORIES))
    return queries.list_knowledge(status, limit, category)


@app.get("/knowledge/{knowledge_id}")
def knowledge_detail(knowledge_id: int):
    result = queries.get_knowledge(knowledge_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Knowledge item not found.")
    return result


@app.get("/review")
def review(status: str = "pending"):
    return queries.list_review(status)


@app.post("/review/{review_id}/resolve")
def resolve_review(review_id: int, body: ReviewResolution):
    with get_connection() as conn:
        item = conn.execute("SELECT type, status FROM review_queue WHERE id = ?",
                            (review_id,)).fetchone()
    if item is None:
        raise HTTPException(status_code=404, detail="Review item not found.")
    if item["status"] != "pending":
        raise HTTPException(status_code=409, detail="Review item already resolved.")

    try:
        if item["type"] == "fuse":
            if body.action not in ("accept", "reject"):
                raise ValueError("action must be 'accept' or 'reject'")
            return resolve_fuse_review(review_id, body.action == "accept")
        if item["type"] == "contradict":
            return resolve_contradiction_review(review_id, body.action)
        if item["type"] == "stabilize":
            result = resolve_stale_review(review_id, body.action)
            if body.action == "confirm":
                try:
                    fusion, lifecycle = integrate_fragment(result["fragment_id"],
                                                           judge=get_judge())
                except Exception:
                    raise HTTPException(status_code=503, detail=PROCESSING_ERROR)
                result.update({"fusion": fusion, "lifecycle": lifecycle})
            return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    raise HTTPException(status_code=400, detail="Unsupported review type.")