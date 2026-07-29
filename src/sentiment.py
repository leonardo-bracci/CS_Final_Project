"""
sentiment.py
------------
Text-emotion component (text stage of the pipeline).

Wraps j-hartmann/emotion-english-distilroberta-base behind a small class,
following the same pattern as Transcriber: the model is loaded once at
construction and reused, so the comparatively slow load cost is paid a single
time rather than per call.

j-hartmann was chosen over VADER and tabularisai/multilingual-sentiment-analysis
after benchmarking all three on a fixed set of test transcripts (see
sentiment_benchmark.py / test_transcripts.py). It was the most accurate on the
test set (6/10 vs VADER and tabularisai) and, crucially, shares the same
seven-emotion label space as the DeepFace facial-emotion model (anger, disgust,
fear, joy, neutral, sadness, surprise) - this lets facial and verbal emotion be
compared directly downstream, e.g. "your face showed sadness while your words
expressed joy", which a plain positive/negative/neutral model can't support.
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
