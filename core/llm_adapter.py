import json

import ollama

from core.intents import CATEGORIES, REQUIRED_SLOTS, validate_proposal


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


SYSTEM_PROMPT = f"""You analyze short personal voice notes.
Return ONLY a JSON object with these keys:
- category: one of {CATEGORIES}
- intent: one of {list(REQUIRED_SLOTS)}
- summary: one short sentence
- slots: object with keys who, what, when, where. Use null for anything not stated.
- confidence: number from 0 to 1

Intent meanings:
- call_person: a call or phone conversation with a specific person, including changing or moving such a call
- schedule_meeting: a meeting or appointment with someone, including changing or moving it
- buy_item: purchasing or getting something
- do_task: work or a chore the user must do or finish (writing, studying, cleaning, submitting, finishing something). The thing to do goes in "what". It does NOT need a person.
- record_idea: a thought, suggestion, or idea to remember
- journal_entry: feelings or events from the user's day
- other: only if none of the above fit

Rules: never invent details that are not in the note. If a person or time is not stated, use null. Only use call_person or schedule_meeting when the note mentions calling, phoning, meeting, or an appointment.

Examples:
Note: "Call Mom tomorrow evening."
{{"category":"reminder","intent":"call_person","summary":"Call Mom tomorrow evening","slots":{{"who":"Mom","what":null,"when":"tomorrow evening","where":null}},"confidence":0.9}}

Note: "We should add dark mode to the app."
{{"category":"idea","intent":"record_idea","summary":"Add dark mode to the app","slots":{{"who":null,"what":"dark mode","when":null,"where":null}},"confidence":0.9}}

Note: "Remind me to call the bank."
{{"category":"reminder","intent":"call_person","summary":"Call the bank","slots":{{"who":"the bank","what":null,"when":null,"where":null}},"confidence":0.9}}

Note: "Actually, move the call with Sam to Thursday."
{{"category":"reminder","intent":"call_person","summary":"Call with Sam moved to Thursday","slots":{{"who":"Sam","what":null,"when":"Thursday","where":null}},"confidence":0.9}}

Note: "I have to finish the lab record by Sunday."
{{"category":"task","intent":"do_task","summary":"Finish the lab record by Sunday","slots":{{"who":null,"what":"lab record","when":"Sunday","where":null}},"confidence":0.9}}

Note: "I need to clean my room."
{{"category":"task","intent":"do_task","summary":"Clean the room","slots":{{"who":null,"what":"clean the room","when":null,"where":null}},"confidence":0.9}}"""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": CATEGORIES},
        "intent": {"type": "string", "enum": list(REQUIRED_SLOTS)},
        "summary": {"type": "string"},
        "slots": {
            "type": "object",
            "properties": {k: {"type": ["string", "null"]} for k in ["who", "what", "when", "where"]},
        },
        "confidence": {"type": "number"},
    },
    "required": ["category", "intent", "summary", "slots", "confidence"],
}


class OllamaLLMAdapter(LLMAdapter):
    def __init__(self, model: str = "qwen2.5:3b", retries: int = 1):
        self.model = model
        self.retries = retries

    def analyze(self, text: str, context: list | None = None) -> dict:
        for _ in range(self.retries + 1):
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": text},
                    ],
                    format=OUTPUT_SCHEMA,
                    options={"temperature": 0},
                )
                raw = json.loads(response["message"]["content"])
                return validate_proposal(raw)
            except (ValueError, KeyError):
                continue  # bad output: try again

        # Safe fallback: never crash the pipeline because of a bad model reply.
        return {
            "category": "idea",
            "intent": "other",
            "summary": text[:80],
            "slots": {"who": None, "what": None, "when": None, "where": None},
            "missing": [],
            "confidence": 0.0,
        }