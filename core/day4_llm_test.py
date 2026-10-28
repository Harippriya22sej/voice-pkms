import time
from core.llm_adapter import OllamaLLMAdapter

notes = [
    "Remind me to call Ravi about the project demo on Friday at 5.",
    "Remind me to call Ravi about the demo.",
    "Buy vegetables from the market tomorrow.",
    "What if we let users search their notes by voice?",
]

llm = OllamaLLMAdapter()
for note in notes:
    start = time.time()
    result = llm.analyze(note)
    print(f"\n{note}\n  ({time.time() - start:.1f}s)")
    print(" ", result)