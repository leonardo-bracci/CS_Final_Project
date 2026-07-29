"""
profile_generator.py
---------------------
Local-language-model component (language stage of the pipeline).

Wraps Ollama behind a small class. Takes the outputs of Stages 1-3 (per-scene
facial emotions, transcribed speech, and text-emotion classification) and:
  - generates an initial psychological reflection + wellbeing recommendations
  - flags whether the response suggests professional help should be sought
  - supports a follow-up interactive chat, using the profile as context

Ollama was chosen because it runs entirely locally - no cloud service, no data
leaves the machine - which is the project's core privacy requirement (see
Design chapter). It also supports small models (2-4GB) that run acceptably on
CPU-only consumer hardware.

IMPORTANT: this component never diagnoses. The system prompt explicitly
forbids clinical/diagnostic language and instructs the model to recommend
professional help when the observed patterns suggest significant distress.
"""

from collections import namedtuple

Profile = namedtuple("Profile", ["summary", "seek_help", "raw_text"])

SYSTEM_PROMPT = """You are a supportive, privacy-preserving wellbeing assistant.
You are NOT a therapist and must NEVER diagnose a medical or psychiatric
condition. Your job is to reflect back the emotional patterns you observe
(facial expression, spoken tone, and the words used) in a warm, non-clinical
way, and suggest general wellbeing practices (e.g. journaling, breathing
exercises, talking to someone they trust). If the observed patterns suggest
significant or persistent distress, gently and clearly recommend the user
speak to a mental health professional, and make clear this tool is not a
substitute for professional care. Keep responses concise and conversational."""

# Crude keyword check for whether the model's reply recommended professional
# help. Not a substitute for the model actually being instructed to do this
# (see SYSTEM_PROMPT) - this is a secondary signal used to flag the session
# for the UI layer (e.g. to surface a resources link).
HELP_KEYWORDS = ["professional", "therapist", "counsellor", "counselor", "seek help", "seek support"]


class ProfileGenerator:
    """Reusable wrapper around a local Ollama model, with conversation history
    kept on the instance so generate_profile() and chat() share context."""

    def __init__(self, model_name="llama3.2"):
        """Import here so the module is importable without the ollama package
        installed, matching the lazy-import pattern used elsewhere in the
        pipeline (DeepFace, faster-whisper, transformers)."""
        import ollama
        self._ollama = ollama
        self.model_name = model_name
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _build_prompt(self, emotion_summary, transcript, text_emotion):
        """Assemble the structured prompt from all three upstream stages into
        one message the model can reason over as a single picture of the user."""
        scene_lines = "\n".join(
            f"- {scene}: {emotion} (dominant facial expression, {count} readings)"
            for scene, (emotion, count) in emotion_summary.items()
        )
        return (
            "Here is what was observed during the session:\n\n"
            f"Facial emotions per scene:\n{scene_lines}\n\n"
            f"What the user said: \"{transcript}\"\n"
            f"Emotional tone of their words: {text_emotion.label} "
            f"(confidence {text_emotion.score})\n\n"
            "Based on this, write a short, warm psychological reflection: "
            "(1) a summary of the emotional patterns you notice, including any "
            "mismatch between facial expression and spoken tone, "
            "(2) one or two focus areas the user might benefit from working on, "
            "(3) two or three concrete, general wellbeing recommendations. "
            "If the patterns suggest significant distress, say so clearly and "
            "recommend professional support."
        )

    def generate_profile(self, emotion_summary, transcript, text_emotion):
        """Generate the initial profile from Stage 1-3 outputs. Returns a
        Profile with the reply text and a seek_help flag."""
        prompt = self._build_prompt(emotion_summary, transcript, text_emotion)
        self.history.append({"role": "user", "content": prompt})

        response = self._ollama.chat(model=self.model_name, messages=self.history)
        reply = response["message"]["content"]
        self.history.append({"role": "assistant", "content": reply})

        seek_help = any(kw in reply.lower() for kw in HELP_KEYWORDS)
        return Profile(summary=reply, seek_help=seek_help, raw_text=reply)

    def chat(self, user_message):
        """Continue the conversation, keeping full history for context."""
        self.history.append({"role": "user", "content": user_message})
        response = self._ollama.chat(model=self.model_name, messages=self.history)
        reply = response["message"]["content"]
        self.history.append({"role": "assistant", "content": reply})
        return reply


if __name__ == "__main__":
    # Minimal smoke test using placeholder Stage 1-3 outputs, so this file can
    # be run on its own without a live webcam/mic session.
    from collections import namedtuple as _nt
    EmotionResult = _nt("EmotionResult", ["label", "score"])

    sample_summary = {
        "threatening scene": ("fear", 5),
        "joyful scene": ("happy", 8),
        "sad scene": ("sad", 6),
        "relaxing scene": ("neutral", 10),
    }
    sample_transcript = "I've been feeling pretty stressed about work lately, but today was okay."
    sample_text_emotion = EmotionResult(label="fear", score=0.6)

    generator = ProfileGenerator(model_name="llama3.2")
    profile = generator.generate_profile(sample_summary, sample_transcript, sample_text_emotion)
    print(profile.summary)
    print("\nSeek-help flag:", profile.seek_help)

    # Try one follow-up turn to confirm the chat loop keeps context
    follow_up = generator.chat("What's one small thing I could try this week?")
    print("\n--- follow-up ---")
    print(follow_up)
