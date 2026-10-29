from core.database import get_connection, init_db
from core.embeddings import embed
from core.fusion import fuse_fragment
from core.intents import validate_proposal
from core.llm_adapter import OllamaLLMAdapter
from core.stabilization import stabilize


def process_text_note(text: str, llm=None) -> dict:
    init_db()
    text = text.strip()
    if not text:
        raise ValueError("Empty note.")
    llm = llm or OllamaLLMAdapter()

    # Save the original first, so it is never lost if analysis fails.
    with get_connection() as conn:
        note_id = conn.execute(
            "INSERT INTO notes (text, source) VALUES (?, 'text')", (text,)
        ).lastrowid

    proposal = validate_proposal(llm.analyze(text))   # model output is untrusted
    result = stabilize(note_id, proposal, embed(text))

    # Fuse when this note produced (or completed) a stable fragment.
    target_id = None
    if result["stability"] == "stable":
        target_id = result["fragment_id"]
    elif result["stability"] == "absorbed" and result["target_stability"] == "stable":
        target_id = result["merged_into"]
    fusion = fuse_fragment(target_id) if target_id else None

    with get_connection() as conn:
        conn.execute("UPDATE notes SET status = 'processed' WHERE id = ?", (note_id,))
    return {"note_id": note_id, **result, "fusion": fusion}