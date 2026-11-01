class SpeechAdapter:
    """Interface every speech-to-text backend must follow."""

    def transcribe(self, audio_path) -> str:
        raise NotImplementedError


class FasterWhisperAdapter(SpeechAdapter):
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        return self._model

    def transcribe(self, audio_path) -> str:
        segments, _info = self._load().transcribe(str(audio_path))
        return " ".join(s.text.strip() for s in segments).strip()