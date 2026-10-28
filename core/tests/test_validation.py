import pytest
from core.intents import validate_proposal
from core.llm_adapter import MockLLMAdapter


def test_bad_category_rejected():
    with pytest.raises(ValueError):
        validate_proposal({"category": "banana", "intent": "buy_item"})


def test_null_strings_become_none_and_missing_is_computed():
    result = validate_proposal({
        "category": "reminder", "intent": "call_person",
        "slots": {"who": "Ravi", "when": "null"}, "confidence": 0.8,
    })
    assert result["slots"]["when"] is None
    assert result["missing"] == ["when"]


def test_confidence_is_clamped():
    result = validate_proposal({"category": "idea", "intent": "record_idea", "confidence": 7})
    assert result["confidence"] == 1.0


def test_unknown_intent_becomes_other():
    # Use a category with no repair rule, so "other" is preserved.
    result = validate_proposal({"category": "task", "intent": "dance"})
    assert result["intent"] == "other"


def test_unknown_intent_on_idea_is_repaired():
    result = validate_proposal({"category": "idea", "intent": "dance"})
    assert result["intent"] == "record_idea"

def test_mock_output_passes_validation():
    proposal = MockLLMAdapter().analyze("Remind me to call Ravi")
    assert validate_proposal(proposal)["missing"] == ["when"]

def test_idea_with_other_intent_is_repaired():
    result = validate_proposal({"category": "idea", "intent": "other"})
    assert result["intent"] == "record_idea"