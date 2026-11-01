import json
import re

from core.database import get_connection
from core.embeddings import cosine, embed, from_blob

W_SEMANTIC = 0.7
W_KEYWORD = 0.3
MIN_SCORE = 0.25

STOPWORDS = {"the", "and", "for", "with", "when", "what", "who", "where", "how", "does",
             "did", "are", "was", "were", "this", "that", "have", "has", "about", "from",
             "into", "any", "can", "you", "your", "all", "not", "but", "its", "get"}


def _tokens(text: str) -> set:
    return {w for w in re.findall(r"\w+", text.lower())
            if len(w) > 2 and w not in STOPWORDS}


def _searchable_text(title: str, content: dict) -> str:
    parts = [title or ""]
    parts += [str(v) for v in content.get("slots", {}).values() if v]
    parts += content.get("summaries", [])
    return " ".join(parts)


def search(query: str, limit: int = 5, include_superseded: bool = False,
           min_score: float = MIN_SCORE) -> list:
    query = query.strip()
    if not query:
        raise ValueError("Empty query.")
    limit = max(1, min(limit, 50))

    qvec = embed(query)
    qtokens = _tokens(query)

    sql = ("SELECT id, category, title, content_json, embedding, status, version, updated_at "
           "FROM knowledge WHERE embedding IS NOT NULL")
    if not include_superseded:
        sql += " AND status = 'active'"
    with get_connection() as conn:
        rows = conn.execute(sql).fetchall()

    results = []
    for r in rows:
        content = json.loads(r["content_json"] or "{}")
        semantic = max(0.0, cosine(qvec, from_blob(r["embedding"])))
        text_tokens = _tokens(_searchable_text(r["title"], content))
        keyword = len(qtokens & text_tokens) / len(qtokens) if qtokens else 0.0
        score = W_SEMANTIC * semantic + W_KEYWORD * keyword
        if score < min_score:
            continue
        results.append({
            "id": r["id"], "title": r["title"], "category": r["category"],
            "status": r["status"], "version": r["version"],
            "slots": content.get("slots", {}), "updated_at": r["updated_at"],
            "score": round(score, 3), "semantic": round(semantic, 3),
            "keyword": round(keyword, 3),
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]