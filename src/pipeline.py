"""
pipeline.py
-----------
Orchestrator for the Emotional AI Mirror.

Runs each model component in turn and passes data between them. The current
prototype uses the webcam-based vision stage plus a sample WAV file from
``samples/`` for the audio stage so the demo can run without manual file
selection.
"""

from pathlib import Path

try:
    from src import emotion_detector
    from src.transcriber import Transcriber
except ImportError:
    import emotion_detector
    from transcriber import Transcriber


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIO_PATH = PROJECT_ROOT / "samples" / "sample.wav"


def run_pipeline(audio_path=DEFAULT_AUDIO_PATH):
    """Run the full analysis pipeline end to end.

    Step 1 (vision): capture a webcam session and detect facial emotion,
    returning a timeline of (time, scene, emotion) readings.
    """
    print("=== Stage 1: facial emotion detection ===")
    timeline = emotion_detector.run_session()

    if not timeline:
        print("No emotion data captured. Ending pipeline.")
        return

    # Reduce the raw timeline to the dominant emotion per scene - this summary
    # is what later stages (sentiment, Ollama) will build on.
    summary = emotion_detector.summarise_timeline(timeline)

    print("\n=== Session summary ===")
    for scene, (emotion, count) in summary.items():
        print(f"{scene}: dominant emotion was {emotion} ({count} readings)")

    print("\n=== Stage 2: sample audio transcription ===")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    transcriber = Transcriber(model_size="tiny")
    transcription = transcriber.transcribe(str(audio_path))
    print(f"[{transcription.language}] {transcription.text}")

    # Future stages will go here:
    #   Stage 3 (text):     sentiment analysis of the transcript
    #   Stage 4 (language): feed everything to Ollama for the profile

    return {
        "emotion_summary": summary,
        "transcription": transcription,
        "audio_path": str(audio_path),
    }


if __name__ == "__main__":
    run_pipeline()