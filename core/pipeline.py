from core.database import get_connection, init_db
from core.embeddings import embed
from core.fusion import fuse_fragment
from core.intents import validate_proposal
from core.lifecycle import OllamaLifecycleJudge, resolve_conflict
from core.llm_adapter import OllamaLLMAdapter
from core.stabilization import stabilize


def integrate_fragment(fragment_id: int, judge=None):
    """Put one stable fragment into the knowledge base (fusion, then lifecycle)."""
    fusion = fuse_fragment(fragment_id)
    lifecycle = None
    if fusion["decision"] == "conflict":
        lifecycle = resolve_conflict(fragment_id, fusion["knowledge_id"],
                                     judge=judge or OllamaLifecycleJudge())
    return fusion, lifecycle


def process_text_note(text: str, llm=None, judge=None,
                      source: str = "text", audio_path: str | None = None) -> dict:
    init_db()
    text = text.strip()
    if not text:
        raise ValueError("Empty note.")
    llm = llm or OllamaLLMAdapter()

    # Save the original first, so it is never lost if analysis fails.
    with get_connection() as conn:
        note_id = conn.execute(
            "INSERT INTO notes (text, source, audio_path) VALUES (?, ?, ?)",
            (text, source, audio_path),
        ).lastrowid

    try:
        proposal = validate_proposal(llm.analyze(text))   # model output is untrusted
        result = stabilize(note_id, proposal, embed(text))

        # A stable fragment (or a pending one this note just completed) goes to knowledge.
        target_id = None
        if result["stability"] == "stable":
            target_id = result["fragment_id"]
        elif result["stability"] == "absorbed" and result["target_stability"] == "stable":
            target_id = result["merged_into"]

        fusion = lifecycle = None
        if target_id:
            fusion, lifecycle = integrate_fragment(target_id, judge)
    except Exception:
        with get_connection() as conn:
            conn.execute("UPDATE notes SET status = 'failed' WHERE id = ?", (note_id,))
        raise

    with get_connection() as conn:
        conn.execute("UPDATE notes SET status = 'processed' WHERE id = ?", (note_id,))
    return {"note_id": note_id, **result, "fusion": fusion, "lifecycle": lifecycle}