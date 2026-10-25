import time
from pathlib import Path
from faster_whisper import WhisperModel

AUDIO = Path("data") / "test1.wav"   # change the name if yours differs

print("Loading model (first run downloads about 150 MB)...")
model = WhisperModel("base", device="cpu", compute_type="int8")

start = time.time()
segments, info = model.transcribe(str(AUDIO))

# transcribe() returns a generator, so the work happens while we loop over it
text = " ".join(segment.text.strip() for segment in segments)

print("Detected language:", info.language)
print("Transcript:", text)
print(f"Took {time.time() - start:.1f} seconds")