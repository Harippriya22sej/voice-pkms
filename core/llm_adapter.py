class LLMAdapter:
    """Interface every LLM backend must follow."""

    def analyze(self, text: str, context: list | None = None) -> dict:
        """Return a structured proposal. Must never write to the database."""
        raise NotImplementedError


class MockLLMAdapter(LLMAdapter):
    """Rule-based fake LLM for development and tests."""

    def analyze(self, text: str, context: list | None = None) -> dict:
        lower = text.lower()
        if "remind" in lower or "call" in lower:
            category, intent, required = "reminder", "call_person", ["who", "when"]
        elif "buy" in lower:
            category, intent, required = "task", "buy_item", ["what"]
        else:
            category, intent, required = "idea", "record_idea", []

        slots = {name: None for name in required}
        if "friday" in lower:
            slots["when"] = "Friday"
        if "ravi" in lower:
            slots["who"] = "Ravi"
        if "vegetables" in lower:
            slots["what"] = "vegetables"

        missing = [k for k, v in slots.items() if v is None]
        return {
            "category": category,
            "intent": intent,
            "summary": text[:80],
            "slots": slots,
            "missing": missing,
            "confidence": 0.9 if not missing else 0.6,
        }