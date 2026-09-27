"""
test_profile_generator.py
-------------------------
Unit tests for profile_generator.py. Ollama is mocked, so the tests run
offline without a real model. They check prompt building, the contact-detail
filter, chat history, and that the model's reply never sets seek_help.
The quality of the model's actual replies is tested separately with the
real model (scripts/ollama_benchmark.py, scripts/chat_robustness_test.py).
"""

import sys
from collections import namedtuple
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from profile_generator import ProfileGenerator, SYSTEM_PROMPT

EmotionResult = namedtuple("EmotionResult", ["label", "score"])


def fake_chat_response(content):
    """Build a response shaped like ollama.chat()'s real return value."""
    return {"message": {"content": content}}


class TestConstruction:
    def test_system_prompt_set_as_first_history_entry(self):
        with patch("ollama.chat"):
            generator = ProfileGenerator(model_name="llama3.2")
            assert generator.history[0] == {"role": "system", "content": SYSTEM_PROMPT}
            assert len(generator.history) == 1


class TestPromptAssembly:
    def test_build_prompt_includes_all_three_stage_outputs(self):
        with patch("ollama.chat"):
            generator = ProfileGenerator()
            summary = {"joyful scene": ("happy", 8), "sad scene": ("sad", 3)}
            transcript = "Today was a pretty good day overall."
            text_emotion = EmotionResult(label="joy", score=0.9)

            prompt = generator._build_prompt(summary, transcript, text_emotion)

            assert "joyful scene" in prompt
            assert "happy" in prompt
            assert "sad scene" in prompt
            assert transcript in prompt
            assert "joy" in prompt
            assert "0.9" in prompt

    def test_build_prompt_handles_empty_summary_and_transcript(self):
        """An empty scene summary or transcript is a valid pipeline state
        (e.g. no webcam readings, silent response), not an error - the
        prompt should still assemble without raising."""
        with patch("ollama.chat"):
            generator = ProfileGenerator()
            prompt = generator._build_prompt({}, "", EmotionResult(label="neutral", score=0.0))

            assert "What the user said" in prompt
            assert "neutral" in prompt


class TestSeekHelpFlag:
    def test_model_reply_no_longer_sets_seek_help(self):
        """Regression test for the pilot finding: a reply mentioning
        'professional' used to set seek_help=True in every session. The flag
        is now assessed later from the user's own words (safety.py), so the
        model's wording must not influence it."""
        with patch("ollama.chat") as mock_chat:
            mock_chat.return_value = fake_chat_response(
                "This tool is not a substitute for professional care."
            )
            generator = ProfileGenerator()
            profile = generator.generate_profile(
                {"joyful scene": ("happy", 8)}, "I feel great", EmotionResult("joy", 0.9)
            )
            assert profile.seek_help is None


class TestChatHistory:
    def test_chat_returns_reply_and_appends_to_history(self):
        with patch("ollama.chat") as mock_chat:
            mock_chat.return_value = fake_chat_response("Sure, here's a suggestion for this week.")
            generator = ProfileGenerator()

            reply = generator.chat("What's one small thing I could try?")

            assert reply == "Sure, here's a suggestion for this week."
            assert generator.history[-2] == {"role": "user", "content": "What's one small thing I could try?"}
            assert generator.history[-1] == {"role": "assistant", "content": "Sure, here's a suggestion for this week."}

    def test_generate_profile_then_chat_share_history(self):
        """A follow-up chat() call should see the profile generated earlier
        in the same session, since both operate on the same history list."""
        with patch("ollama.chat") as mock_chat:
            mock_chat.side_effect = [
                fake_chat_response("Here is your profile."),
                fake_chat_response("Following up on your profile, here's more detail."),
            ]
            generator = ProfileGenerator()

            generator.generate_profile({}, "some text", EmotionResult("neutral", 0.5))
            generator.chat("Tell me more")

            # system + profile prompt + profile reply + follow-up + follow-up reply
            assert len(generator.history) == 5
            assert generator.history[1]["role"] == "user"
            assert generator.history[2] == {"role": "assistant", "content": "Here is your profile."}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestContactDetailFilter:
    def test_phone_numbers_short_codes_urls_and_placeholders_are_removed(self):
        from src.profile_generator import strip_contact_details
        reply = (
            "Please reach out.\n"
            "1. Lifeline: 1-800-273-TALK (8255)\n"
            "2. Crisis Text Line: Text HOME to 741741\n"
            "* NAMI Helpline: [insert NAMI helpline number]\n"
            "Visit https://example.org for more.\n"
            "Try the 4-7-8 breathing exercise.\n"
            "You deserve support."
        )
        assert strip_contact_details(reply) == (
            "Please reach out.\nTry the 4-7-8 breathing exercise.\nYou deserve support."
        )

    def test_chat_reply_is_filtered_before_being_returned_and_stored(self):
        with patch("ollama.chat") as mock_chat:
            mock_chat.return_value = fake_chat_response("You matter.\nCall 1-800-273-8255 now.")
            generator = ProfileGenerator()
            reply = generator.chat("I feel sad")
            assert reply == "You matter."
            assert generator.history[-1]["content"] == "You matter."
