"""
test_session.py
---------------
Unit tests for session.py. The microphone and keyboard input are mocked,
so the tests run offline with no one at the keyboard. They check the consent
gate, question and answer collection, transcript formatting and the closing
message.
"""

import sys
from collections import namedtuple
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import session
from session import (
    SPOKEN_QUESTIONS,
    format_qa_transcript,
    get_consent,
    print_closing,
    run_spoken_questions,
)

Profile = namedtuple("Profile", ["summary", "seek_help", "raw_text"])
Transcription = namedtuple("Transcription", ["text", "language", "duration"])


class TestGetConsent:
    def test_auto_yes_skips_prompt_and_returns_true(self):
        with patch("builtins.input") as mock_input:
            result = get_consent(auto_yes=True)
            assert result is True
            mock_input.assert_not_called()

    def test_typed_yes_returns_true(self):
        with patch("builtins.input", return_value="yes"):
            assert get_consent(auto_yes=False) is True

    def test_typed_yes_is_case_insensitive_and_strips_whitespace(self):
        with patch("builtins.input", return_value="  YES  "):
            assert get_consent(auto_yes=False) is True

    def test_blank_input_is_treated_as_declined(self):
        with patch("builtins.input", return_value=""):
            assert get_consent(auto_yes=False) is False

    def test_explicit_no_is_declined(self):
        with patch("builtins.input", return_value="no"):
            assert get_consent(auto_yes=False) is False

    def test_eof_is_treated_as_declined_not_an_error(self):
        """A non-interactive context (e.g. a scripted run with no stdin)
        must fail closed - consent withheld, not an exception."""
        with patch("builtins.input", side_effect=EOFError):
            assert get_consent(auto_yes=False) is False

    def test_keyboard_interrupt_is_treated_as_declined(self):
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            assert get_consent(auto_yes=False) is False


class TestRunSpokenQuestions:
    def test_asks_all_three_questions_in_order_live_mode(self):
        fake_transcriber = MagicMock()
        fake_transcriber.transcribe.side_effect = [
            Transcription(text="answer one", language="en", duration=1.0),
            Transcription(text="answer two", language="en", duration=1.0),
            Transcription(text="answer three", language="en", duration=1.0),
        ]

        # record_audio is imported lazily inside the function as
        # `from recorder import record_audio`, so it's patched at its
        # source module (recorder), not on session.
        with patch("recorder.record_audio", return_value=Path("/tmp/q.wav")) as mock_record:
            qa_pairs = run_spoken_questions(
                fake_transcriber, use_sample=False, sample_audio_path=None, mic_dir=Path("/tmp")
            )

        assert len(qa_pairs) == 3
        assert [pair["question"] for pair in qa_pairs] == SPOKEN_QUESTIONS
        assert [pair["answer"] for pair in qa_pairs] == ["answer one", "answer two", "answer three"]
        assert mock_record.call_count == 3
        assert fake_transcriber.transcribe.call_count == 3

    def test_demo_mode_reuses_the_same_sample_for_all_three_questions(self):
        fake_transcriber = MagicMock()
        fake_transcriber.transcribe.return_value = Transcription(
            text="sample answer", language="en", duration=1.0
        )
        sample_path = Path("/tmp/sample.wav")

        qa_pairs = run_spoken_questions(
            fake_transcriber, use_sample=True, sample_audio_path=sample_path, mic_dir=Path("/tmp")
        )

        assert len(qa_pairs) == 3
        assert all(pair["answer"] == "sample answer" for pair in qa_pairs)
        for call in fake_transcriber.transcribe.call_args_list:
            assert call.args[0] == str(sample_path)


class TestFormatQaTranscript:
    def test_includes_every_question_and_answer(self):
        qa_pairs = [
            {"question": "How are you feeling right now?", "answer": "Pretty good."},
            {"question": "What did those images bring to mind?", "answer": "My last exam."},
        ]
        transcript = format_qa_transcript(qa_pairs)

        assert "How are you feeling right now?" in transcript
        assert "Pretty good." in transcript
        assert "What did those images bring to mind?" in transcript
        assert "My last exam." in transcript

    def test_empty_list_returns_empty_string(self):
        assert format_qa_transcript([]) == ""

    def test_output_is_deterministically_ordered(self):
        qa_pairs = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
        ]
        transcript = format_qa_transcript(qa_pairs)
        assert transcript.index("Q1") < transcript.index("Q2")


class TestPrintClosing:
    def test_always_prints_resources_text(self, capsys):
        profile = Profile(summary="fine", seek_help=False, raw_text="fine")
        print_closing(profile)
        captured = capsys.readouterr()
        assert "Samaritans" in captured.out
        assert "not a substitute for professional support" in captured.out.replace("\n", " ")

    def test_seek_help_true_adds_stronger_prompt(self, capsys):
        profile = Profile(summary="concerning", seek_help=True, raw_text="concerning")
        print_closing(profile)
        captured = capsys.readouterr()
        assert "good moment to" in captured.out

    def test_seek_help_false_omits_stronger_prompt(self, capsys):
        profile = Profile(summary="fine", seek_help=False, raw_text="fine")
        print_closing(profile)
        captured = capsys.readouterr()
        assert "good moment to" not in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
