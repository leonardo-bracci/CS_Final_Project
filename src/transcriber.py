"""
transcriber.py
--------------
Speech-to-text component (audio stage of the pipeline).

Wraps faster-whisper behind a small class. faster-whisper is used over the
original openai-whisper because it runs ~4x faster on CPU at the same accuracy,
and int8 quantisation keeps memory low - both matter given the project runs
locally with no GPU. The model is loaded once at construction and reused, so
the slow load cost is paid a single time rather than per file.
"""

from pathlib import Path
from collections import namedtuple

# Simple structured result. Keeps the return type explicit: downstream stages
# use .text, benchmarking uses .language / .duration.
Transcription = namedtuple("Transcription", ["text", "language", "duration"])


class Transcriber:
    """Reusable wrapper around a faster-whisper model."""

    def __init__(self, model_size="base"):
        """Load the model once. Imported here (not at top of file) so the module
        can be imported without faster-whisper installed - only building a
        Transcriber actually needs it."""
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, audio_path, language="en"):
        """Transcribe an audio file and return a Transcription.

        faster-whisper returns segments as a generator, so nothing runs until we
        iterate it; we join the segment texts into one clean string for the
        downstream sentiment stage.
        """
        segments, info = self.model.transcribe(audio_path, language=language, beam_size=5)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return Transcription(text=text, language=info.language, duration=info.duration)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    sample_audio = project_root / "samples" / "sample.wav"
    t = Transcriber()
    result = t.transcribe(str(sample_audio))
    print(f"[{result.language}] {result.text}")