from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core import queries
from core.database import get_connection, init_db
from core.fusion import resolve_fuse_review
from core.lifecycle import OllamaLifecycleJudge, resolve_contradiction_review
from core.llm_adapter import OllamaLLMAdapter
from core.pipeline import integrate_fragment, process_text_note
from core.stabilization import confirm_fragment

# Created lazily; tests replace these with fakes.
state = {"llm": None, "judge": None}


def get_llm():
    if state["llm"] is None:
        state["llm"] = OllamaLLMAdapter()
    return state["llm"]


def get_judge():
    if state["judge"] is None:
        state["judge"] = OllamaLifecycleJudge()
    return state["judge"]


@asynccontextmanager
async def lifespan(app):
    init_db()
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


@app.post("/notes/text")
def create_text_note(note: TextNote):
    try:
        return process_text_note(note.text, llm=get_llm(), judge=get_judge())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=503, detail=PROCESSING_ERROR)


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


@app.get("/knowledge")
def knowledge(status: str = "active", limit: int = 50):
    if status not in ("active", "superseded", "all"):
        raise HTTPException(status_code=400, detail="status must be active, superseded or all")
    return queries.list_knowledge(status, limit)


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
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    raise HTTPException(status_code=400, detail="Unsupported review type.")