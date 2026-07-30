"""
pipeline.py
-----------
Orchestrator for the Emotional AI Mirror.

Runs the guided session end to end: consent, stimulus, three spoken
questions, processing, profile, and interactive follow-up chat, then a
closing message with professional-help resources. By default the audio
stage records live from the microphone; pass --demo to use the bundled
samples/sample.wav instead (for all three questions), which is useful for
offline demos, the Topic 9 peer review, or when a working microphone isn't
available. --demo also auto-confirms consent, since it's intended for
walkthroughs where a real person has already agreed to run the demo out of
band; pass --auto-consent on its own to skip the consent prompt without
switching to sample audio.

Every run is logged to a timestamped JSON file in logs/, capturing consent,
all three spoken Q&A pairs, and every stage's output plus the full
follow-up chat transcript. This is what later becomes the Evaluation
chapter's session data, rather than something copied by hand from the
terminal after each run. Sessions where consent is withheld are not run at
all, and nothing is recorded or logged for them.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from src import emotion_detector
    from src import session
    from src.transcriber import Transcriber
    from src.sentiment import SentimentAnalyzer
    from src.profile_generator import ProfileGenerator
except ImportError:
    import emotion_detector
    import session
    from transcriber import Transcriber
    from sentiment import SentimentAnalyzer
    from profile_generator import ProfileGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIO_PATH = PROJECT_ROOT / "samples" / "sample.wav"
MIC_DIR = PROJECT_ROOT / "samples"
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


def save_session_log(emotion_summary, qa_pairs, text_emotion, profile, generator, use_sample):
    """Write one timestamped JSON file per session, capturing consent, the
    three spoken Q&A pairs, every stage's output, and the full chat
    transcript (excluding the system prompt).

    Namedtuples (EmotionResult, Profile) don't serialise to JSON directly,
    so each is converted via its own _asdict() rather than a generic
    serialiser - this keeps the log's shape explicit and predictable for
    later analysis, rather than depending on json's default() fallback.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # generator.history includes the system prompt as its first entry, which
    # is fixed and identical across every session - excluded here since it
    # adds nothing to a per-session log and would just be repeated noise.
    chat_history = [msg for msg in generator.history if msg["role"] != "system"]

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "consent_given": True,
        "demo_mode": use_sample,
        "emotion_summary": {
            scene: {"emotion": emotion, "readings": count}
            for scene, (emotion, count) in emotion_summary.items()
        },
        "spoken_qa": qa_pairs,
        "text_emotion": text_emotion._asdict(),
        "profile": profile._asdict(),
        "chat_history": chat_history,
    }

    log_path = LOGS_DIR / f"session_{log_entry['timestamp'].replace(':', '-')}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_entry, f, indent=2, ensure_ascii=False)

    print(f"\nSession log saved to {log_path}")
    return log_path


def run_pipeline(use_sample=False, audio_path=DEFAULT_AUDIO_PATH, auto_consent=False):
    """Run the full guided session: consent, stimulus, three spoken
    questions, the four-stage analysis pipeline, profile generation,
    interactive follow-up chat, and a closing message with resources. The
    full session is logged to logs/ once the chat ends.

    use_sample=True skips live microphone recording and transcribes the
    bundled sample.wav for all three questions instead - used for --demo
    mode and for testing without a working microphone. auto_consent=True
    (set automatically by --demo, or independently by --auto-consent) skips
    the interactive consent prompt.

    Returns None if consent is withheld - no stage is run and nothing is
    recorded or logged for a declined session.
    """
    print("=== Consent ===")
    consented = session.get_consent(auto_yes=auto_consent or use_sample)
    if not consented:
        print("Consent not given - ending session. Nothing was recorded.")
        return None

    print("\n=== Stimulus: watch the sequence and let your face react naturally ===")
    timeline = emotion_detector.run_session()

    if not timeline:
        print("No emotion data captured. Ending pipeline.")
        return None

    summary = emotion_detector.summarise_timeline(timeline)

    print("\n=== Session summary ===")
    for scene, (emotion, count) in summary.items():
        print(f"{scene}: dominant emotion was {emotion} ({count} readings)")

    print("\n=== Spoken questions ===")
    transcriber = Transcriber(model_size="tiny")
    qa_pairs = session.run_spoken_questions(
        transcriber, use_sample=use_sample, sample_audio_path=audio_path, mic_dir=MIC_DIR
    )
    combined_transcript = session.format_qa_transcript(qa_pairs)

    print("\n=== Text-emotion analysis ===")
    analyzer = SentimentAnalyzer()
    text_emotion = analyzer.analyze(combined_transcript)
    print(f"[{text_emotion.label} ({text_emotion.score})]")

    print("\n=== Psychological profile generation (Ollama) ===")
    generator = ProfileGenerator(model_name="llama3.2")
    profile = generator.generate_profile(summary, combined_transcript, text_emotion)
    print(profile.summary)
    print(f"\nSeek-help flag: {profile.seek_help}")

    run_chat_loop(generator)

    session.print_closing(profile)

    save_session_log(summary, qa_pairs, text_emotion, profile, generator, use_sample)

    return {
        "emotion_summary": summary,
        "spoken_qa": qa_pairs,
        "text_emotion": text_emotion,
        "profile": profile,
    }


if __name__ == "__main__":
    demo_mode = "--demo" in sys.argv
    auto_consent_flag = "--auto-consent" in sys.argv
    run_pipeline(use_sample=demo_mode, auto_consent=auto_consent_flag)
