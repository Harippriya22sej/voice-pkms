from core.pipeline import process_text_note

notes = [
    "Remind me to call Ravi about the demo.",       # missing "when" -> pending
    "The call with Ravi is on Friday at 5.",         # should fill the gap
    "Buy vegetables from the market tomorrow.",      # complete -> stable
]

for text in notes:
    print("\n", text)
    print("  ->", process_text_note(text))