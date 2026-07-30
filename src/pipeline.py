"""
pipeline.py
-----------
Orchestrator for the Emotional AI Mirror.

Runs each model component in turn and passes data between them. By default
the audio stage records live from the microphone; pass --demo to use the
bundled samples/sample.wav instead, which is useful for offline demos or
when a working microphone isn't available.

Every run is logged to a timestamped JSON file in logs/, capturing all four
stages' outputs plus the full follow-up chat transcript. This is what later
becomes the Evaluation chapter's session data, rather than something copied
by hand from the terminal after each run.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from src import emotion_detector
    from src.transcriber import Transcriber
    from src.sentiment import SentimentAnalyzer
    from src.profile_generator import ProfileGenerator
    from src.recorder import record_audio
except ImportError:
    import emotion_detector
    from transcriber import Transcriber
    from sentiment import SentimentAnalyzer
    from profile_generator import ProfileGenerator
    from recorder import record_audio


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIO_PATH = PROJECT_ROOT / "samples" / "sample.wav"
MIC_RECORDING_PATH = PROJECT_ROOT / "samples" / "session_recording.wav"
LOGS_DIR = PROJECT_ROOT / "logs"

# Typing any of these (case-insensitive) ends the follow-up chat session.
QUIT_WORDS = {"quit", "exit", "q", "bye", "stop"}


def run_chat_loop(generator):
    """Interactive follow-up conversation with the language model, reusing
    the same ProfileGenerator instance so context (the profile it just wrote)
    carries into every reply."""
    print("\nYou can now ask questions about your profile.")
    print("Type 'quit' at any time to end the conversation.\n")

    while True:
        try:
            user_message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEnding session.")
            break

        if not user_message:
            continue
        if user_message.lower() in QUIT_WORDS:
            print("Ending session. Take care.")
            break

        reply = generator.chat(user_message)
        print(f"\nAssistant: {reply}\n")


def save_session_log(emotion_summary, transcription, text_emotion, profile, generator, audio_path):
    """Write one timestamped JSON file per session, capturing every stage's
    output plus the full chat transcript (excluding the system prompt).

    Namedtuples (Transcription, EmotionResult, Profile) don't serialise to
    JSON directly, so each is converted via its own _asdict() rather than a
    generic serialiser - this keeps the log's shape explicit and predictable
    for later analysis, rather than depending on json's default() fallback.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # generator.history includes the system prompt as its first entry, which
    # is fixed and identical across every session - excluded here since it
    # adds nothing to a per-session log and would just be repeated noise.
    chat_history = [msg for msg in generator.history if msg["role"] != "system"]

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "audio_path": str(audio_path),
        "emotion_summary": {
            scene: {"emotion": emotion, "readings": count}
            for scene, (emotion, count) in emotion_summary.items()
        },
        "transcription": transcription._asdict(),
        "text_emotion": text_emotion._asdict(),
        "profile": profile._asdict(),
        "chat_history": chat_history,
    }

    log_path = LOGS_DIR / f"session_{log_entry['timestamp'].replace(':', '-')}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_entry, f, indent=2, ensure_ascii=False)

    print(f"\nSession log saved to {log_path}")
    return log_path


def run_pipeline(use_sample=False, audio_path=DEFAULT_AUDIO_PATH):
    """Run the full four-stage analysis pipeline end to end: vision, audio,
    text-emotion, then language-model synthesis into a profile, followed by
    an interactive follow-up chat. The full session is logged to logs/ once
    the chat ends.

    use_sample=True skips live microphone recording and transcribes the
    bundled sample.wav instead - used for --demo mode and for testing
    without a working microphone.
    """
    print("=== Stage 1: facial emotion detection ===")
    timeline = emotion_detector.run_session()

    if not timeline:
        print("No emotion data captured. Ending pipeline.")
        return

    summary = emotion_detector.summarise_timeline(timeline)

    print("\n=== Session summary ===")
    for scene, (emotion, count) in summary.items():
        print(f"{scene}: dominant emotion was {emotion} ({count} readings)")

    print("\n=== Stage 2: speech-to-text ===")
    if use_sample:
        print("(demo mode: using bundled sample.wav instead of live microphone)")
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        recording_path = audio_path
    else:
        recording_path = record_audio(MIC_RECORDING_PATH)

    transcriber = Transcriber(model_size="tiny")
    transcription = transcriber.transcribe(str(recording_path))
    print(f"[{transcription.language}] {transcription.text}")

    print("\n=== Stage 3: text-emotion analysis ===")
    analyzer = SentimentAnalyzer()
    text_emotion = analyzer.analyze(transcription.text)
    print(f"[{text_emotion.label} ({text_emotion.score})] {transcription.text}")

    print("\n=== Stage 4: psychological profile generation (Ollama) ===")
    generator = ProfileGenerator(model_name="llama3.2")
    profile = generator.generate_profile(summary, transcription.text, text_emotion)
    print(profile.summary)
    print(f"\nSeek-help flag: {profile.seek_help}")

    run_chat_loop(generator)

    save_session_log(summary, transcription, text_emotion, profile, generator, recording_path)

    return {
        "emotion_summary": summary,
        "transcription": transcription,
        "text_emotion": text_emotion,
        "profile": profile,
        "audio_path": str(recording_path),
    }


if __name__ == "__main__":
    demo_mode = "--demo" in sys.argv
    run_pipeline(use_sample=demo_mode)
