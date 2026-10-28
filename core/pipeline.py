from core.database import get_connection
from core.embeddings import embed
from core.intents import validate_proposal
from core.llm_adapter import OllamaLLMAdapter
from core.stabilization import stabilize


def process_text_note(text: str, llm=None) -> dict:
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

    with get_connection() as conn:
        conn.execute("UPDATE notes SET status = 'processed' WHERE id = ?", (note_id,))
    return {"note_id": note_id, **result}