"""
sentiment.py
------------
Text stage: wraps j-hartmann/emotion-english-distilroberta-base.

The model is loaded once when the class is created and reused for every
call, so the slow load happens only once.

Chosen over VADER and tabularisai/multilingual-sentiment-analysis after
benchmarking all three on the same 10 test transcripts (see
scripts/sentiment_benchmark.py). It scored best (7/10) and uses the same
seven emotions as DeepFace, so facial and verbal emotion can be compared,
e.g. "your face showed sadness while your words expressed joy".
"""

from collections import namedtuple

# Simple structured result, matching the Transcription namedtuple pattern in
# transcriber.py. label is one of DeepFace's seven emotion classes; score is
# the model's confidence in that label.
EmotionResult = namedtuple("EmotionResult", ["label", "score"])


class SentimentAnalyzer:
    """Reusable wrapper around the j-hartmann emotion-classification model."""

    MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

    def __init__(self):
        """Load the model once. Imported here (not at top of file) so the module
        can be imported without transformers/torch installed - only building a
        SentimentAnalyzer actually needs it. Same lazy-import pattern used for
        DeepFace in emotion_detector.py and faster-whisper in transcriber.py."""
        from transformers import pipeline
        self.classifier = pipeline("text-classification", model=self.MODEL_NAME, top_k=1)

    def analyze(self, text):
        """Classify a piece of text and return an EmotionResult.

        Returns a neutral/0.0 result for empty or whitespace-only input rather
        than calling the model - an empty transcript (e.g. the user said
        nothing, or the audio stage dropped a segment) is a valid pipeline
        state, not an error, and shouldn't crash the run.
        """
        if not text or not text.strip():
            return EmotionResult(label="neutral", score=0.0)

        output = self.classifier(text)
        # top_k=1 returns a nested list like [[{"label": ..., "score": ...}]]
        top = output[0][0] if isinstance(output[0], list) else output[0]
        return EmotionResult(label=top["label"], score=round(top["score"], 3))


if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    samples = [
        "Honestly I feel really good today, things have been going well.",
        "I'm fine.",
        "",  # empty-input edge case
    ]
    for text in samples:
        result = analyzer.analyze(text)
        print(f"[{result.label} ({result.score})] {text!r}")
