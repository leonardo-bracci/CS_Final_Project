"""
test_sentiment.py
-----------------
Unit tests for sentiment.py. The transformers pipeline is mocked, so the
tests run offline without the real model. They check the empty-input guard
and how the model's output is turned into a result. Classification accuracy
is tested separately with the real model (scripts/sentiment_benchmark.py).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Allow running pytest from the project root or from tests/ directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentiment import SentimentAnalyzer, EmotionResult


class TestEmptyInputGuard:
    """analyze() should short-circuit on empty/whitespace input rather than
    calling the (mocked, or real) model at all."""

    def test_empty_string_returns_neutral_zero_score(self):
        with patch("transformers.pipeline") as mock_pipeline_fn:
            fake_classifier = MagicMock()
            mock_pipeline_fn.return_value = fake_classifier

            analyzer = SentimentAnalyzer()
            result = analyzer.analyze("")

            assert result == EmotionResult(label="neutral", score=0.0)
            fake_classifier.assert_not_called()

    def test_whitespace_only_returns_neutral_zero_score(self):
        with patch("transformers.pipeline") as mock_pipeline_fn:
            fake_classifier = MagicMock()
            mock_pipeline_fn.return_value = fake_classifier

            analyzer = SentimentAnalyzer()
            result = analyzer.analyze("   \n\t  ")

            assert result == EmotionResult(label="neutral", score=0.0)
            fake_classifier.assert_not_called()


class TestAnalyzeResultShape:
    """analyze() should extract the top label/score correctly regardless of
    the exact nesting shape the underlying pipeline returns."""

    def test_nested_output_shape(self):
        # top_k=1 normally returns [[{"label": ..., "score": ...}]]
        with patch("transformers.pipeline") as mock_pipeline_fn:
            def fake_classifier(text):
                return [[{"label": "joy", "score": 0.97654}]]
            mock_pipeline_fn.return_value = fake_classifier

            analyzer = SentimentAnalyzer()
            result = analyzer.analyze("I feel great today")

            assert result.label == "joy"
            assert result.score == 0.977  # rounded to 3 decimal places

    def test_flat_output_shape(self):
        # guard against a non-nested [{"label": ..., "score": ...}] shape too
        with patch("transformers.pipeline") as mock_pipeline_fn:
            def fake_classifier(text):
                return [{"label": "sadness", "score": 0.5}]
            mock_pipeline_fn.return_value = fake_classifier

            analyzer = SentimentAnalyzer()
            result = analyzer.analyze("some text")

            assert result.label == "sadness"
            assert result.score == 0.5

    def test_passes_input_text_through_to_classifier(self):
        with patch("transformers.pipeline") as mock_pipeline_fn:
            fake_classifier = MagicMock(return_value=[[{"label": "neutral", "score": 0.8}]])
            mock_pipeline_fn.return_value = fake_classifier

            analyzer = SentimentAnalyzer()
            analyzer.analyze("a specific test sentence")

            fake_classifier.assert_called_once_with("a specific test sentence")


class TestModelConstruction:
    def test_model_loaded_once_at_construction(self):
        """The classifier should be built once in __init__, not per analyze()
        call - this is the whole point of the class-based wrapper."""
        with patch("transformers.pipeline") as mock_pipeline_fn:
            mock_pipeline_fn.return_value = MagicMock(
                return_value=[[{"label": "neutral", "score": 0.5}]]
            )

            analyzer = SentimentAnalyzer()
            analyzer.analyze("first")
            analyzer.analyze("second")
            analyzer.analyze("third")

            assert mock_pipeline_fn.call_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
