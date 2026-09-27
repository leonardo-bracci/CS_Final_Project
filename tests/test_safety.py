"""Unit tests for the end-of-session seek-help assessment (src/safety.py).

A fake analyzer returns preset labels, so these tests check the rule itself
without loading the j-hartmann model."""

from collections import namedtuple

from src.safety import assess_session, negative_scene_ratio

EmotionResult = namedtuple("EmotionResult", ["label", "score"])

NEGATIVE_FACE = {"a": ("sad", 5), "b": ("fear", 4), "c": ("happy", 3), "d": ("neutral", 6)}
POSITIVE_FACE = {"a": ("happy", 5), "b": ("neutral", 4), "c": ("happy", 3), "d": ("neutral", 6)}
NEUTRAL = EmotionResult("neutral", 0.86)


class FakeAnalyzer:
    """Returns results keyed by text; records what it was asked to score."""
    def __init__(self, mapping):
        self.mapping = mapping
        self.seen = []

    def analyze(self, text):
        self.seen.append(text)
        return self.mapping[text]


def test_single_strong_negative_text_flags_despite_neutral_texts():
    """One clear disclosure must flag even when every other text is neutral."""
    analyzer = FakeAnalyzer({"n1": NEUTRAL, "n2": NEUTRAL, "n3": NEUTRAL, "n4": NEUTRAL,
                             "bad": EmotionResult("sadness", 0.95)})
    risk = assess_session(["n1", "n2", "n3", "n4", "bad"], analyzer, POSITIVE_FACE)
    assert risk.flag is True


def test_two_moderate_negative_texts_flag():
    analyzer = FakeAnalyzer({"t1": EmotionResult("fear", 0.75), "t2": EmotionResult("disgust", 0.72),
                             "n1": NEUTRAL, "n2": NEUTRAL, "n3": NEUTRAL})
    assert assess_session(["n1", "n2", "n3", "t1", "t2"], analyzer, POSITIVE_FACE).flag is True


def test_one_moderate_negative_text_does_not_flag():
    analyzer = FakeAnalyzer({"t1": EmotionResult("sadness", 0.80), "n1": NEUTRAL})
    assert assess_session(["t1", "n1"], analyzer, NEGATIVE_FACE).flag is False


def test_anger_and_disgust_count_as_negative():
    analyzer = FakeAnalyzer({"t1": EmotionResult("anger", 0.8), "t2": EmotionResult("disgust", 0.8)})
    assert assess_session(["t1", "t2"], analyzer, POSITIVE_FACE).flag is True


def test_low_confidence_negative_does_not_count():
    analyzer = FakeAnalyzer({"t1": EmotionResult("sadness", 0.5), "t2": EmotionResult("fear", 0.6)})
    assert assess_session(["t1", "t2"], analyzer, NEGATIVE_FACE).flag is False


def test_positive_texts_do_not_flag_even_with_negative_face():
    """Face is logged but not used for the flag."""
    analyzer = FakeAnalyzer({"t1": EmotionResult("joy", 0.95), "t2": EmotionResult("surprise", 0.9)})
    assert assess_session(["t1", "t2"], analyzer, NEGATIVE_FACE).flag is False


def test_regression_session_2026_09_27():
    """Real session that the earlier share-based rule missed (2 of 6 = 33%)."""
    analyzer = FakeAnalyzer({
        "demo": EmotionResult("neutral", 0.862),
        "I feel very sad and depressed": EmotionResult("sadness", 0.988),
        "I feel like hirting myself": EmotionResult("disgust", 0.704),
        "my quit": EmotionResult("sadness", 0.911),
    })
    texts = ["demo", "demo", "demo", "I feel very sad and depressed",
             "I feel like hirting myself", "my quit"]
    assert assess_session(texts, analyzer, NEGATIVE_FACE).flag is True


def test_empty_texts_are_skipped_and_do_not_flag():
    analyzer = FakeAnalyzer({})
    risk = assess_session(["", "   "], analyzer, NEGATIVE_FACE)
    assert risk.flag is False
    assert analyzer.seen == []


def test_every_scored_text_is_recorded_for_the_log():
    analyzer = FakeAnalyzer({"t1": EmotionResult("sadness", 0.9), "t2": EmotionResult("joy", 0.8)})
    risk = assess_session(["t1", "t2"], analyzer, NEGATIVE_FACE)
    assert [s["text"] for s in risk.scored_texts] == ["t1", "t2"]
    assert "not used for the flag" in risk.reason


def test_negative_scene_ratio_empty_summary():
    assert negative_scene_ratio({}) == 0.0
