CATEGORIES = ["task", "reminder", "idea", "journal"]
SLOT_NAMES = ["who", "what", "when", "where"]

# Level 2 (intent) -> Level 3 (required slots). Day 5 uses this to decide "stable".
REQUIRED_SLOTS = {
    "call_person": ["who", "when"],
    "schedule_meeting": ["who", "when"],
    "buy_item": ["what"],
    "record_idea": [],
    "journal_entry": [],
    "other": [],
}

_EMPTY_VALUES = {"", "null", "none", "unknown", "n/a", "not specified"}


def validate_proposal(raw) -> dict:
    """Turn raw model output into a clean proposal, or raise ValueError."""
    if not isinstance(raw, dict):
        raise ValueError("Model output is not a JSON object.")

    category = raw.get("category")
    if category not in CATEGORIES:
        raise ValueError(f"Invalid category: {category!r}")

    intent = raw.get("intent")
    if intent not in REQUIRED_SLOTS:
        intent = "other"

    # Category and intent should agree; repair obvious mismatches.
    if intent == "other":
        if category == "idea":
            intent = "record_idea"
        elif category == "journal":
            intent = "journal_entry"

    raw_slots = raw.get("slots")
        

    raw_slots = raw.get("slots")
    if not isinstance(raw_slots, dict):
        raw_slots = {}
    slots = {}
    for name in SLOT_NAMES:
        value = raw_slots.get(name)
        if isinstance(value, str) and value.strip().lower() not in _EMPTY_VALUES:
            slots[name] = value.strip()
        else:
            slots[name] = None

    try:
        confidence = float(raw.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    summary = raw.get("summary")
    summary = summary.strip()[:200] if isinstance(summary, str) else ""

    # "missing" is computed by us, never trusted from the model.
    missing = [s for s in REQUIRED_SLOTS[intent] if slots[s] is None]

    return {
        "category": category,
        "intent": intent,
        "summary": summary,
        "slots": slots,
        "missing": missing,
        "confidence": confidence,
    }