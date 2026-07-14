"""
pipeline.py
-----------
Orchestrator for the Emotional AI Mirror.

Runs each model component in turn and passes data between them. At this stage
only the vision component (facial emotion detection) is wired in; the audio
(speech-to-text), text (sentiment), and language (Ollama) stages will be added
here as they are built, keeping each model in its own module.
"""

import emotion_detector


def run_pipeline():
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

    # Future stages will go here:
    #   Stage 2 (audio):    transcribe the user's spoken answers (Whisper)
    #   Stage 3 (text):     sentiment analysis of the transcript
    #   Stage 4 (language): feed everything to Ollama for the profile

    return summary


if __name__ == "__main__":
    run_pipeline()