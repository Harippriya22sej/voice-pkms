from core.intents import validate_proposal
from core.llm_adapter import OllamaLLMAdapter

llm = OllamaLLMAdapter()
for text in [
    "Actually, move the Ravi call to Monday.",
    "Actually, move my call with Ravi to Monday.",
    "Change the call with Ravi to Monday.",
]:
    p = validate_proposal(llm.analyze(text))
    print("\n", text)
    print("  intent:", p["intent"], "| slots:", p["slots"], "| missing:", p["missing"])