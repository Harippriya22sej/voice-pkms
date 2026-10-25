from core.llm_adapter import MockLLMAdapter


def test_complete_note_has_no_missing_slots():
    result = MockLLMAdapter().analyze("Call Ravi on Friday")
    assert result["missing"] == []


def test_incomplete_note_reports_missing_slot():
    result = MockLLMAdapter().analyze("Remind me to call Ravi")
    assert result["missing"] == ["when"]