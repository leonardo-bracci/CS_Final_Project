"""
safety.py
---------
End-of-session seek-help assessment.

Scores only the user's own words (the three spoken answers and every chat
message) with the j-hartmann text-emotion model. The language model's
replies are ignored, and the check runs in plain Python after the chat,
so the model cannot set or override the flag.

The flag is set if:
  - any single text is negative (sadness, fear, anger or disgust) with
    confidence >= STRONG_CONFIDENCE_THRESHOLD, or
  - at least MIN_DISTRESS_TEXTS texts are negative with confidence
    >= TEXT_CONFIDENCE_THRESHOLD.

The facial data is logged but not used for the flag. Every text's score
and the reason for the result are saved in the session log.

This is a signposting aid, not a diagnosis or crisis assessment. Support
resources are shown at the end of every session regardless of the flag.
"""

from collections import namedtuple

RiskAssessment = namedtuple(
    "RiskAssessment", ["flag", "reason", "distress_share", "negative_face_ratio", "scored_texts"]
)

DISTRESS_TEXT_LABELS = {"sadness", "fear", "anger", "disgust"}  # j-hartmann label space
NEGATIVE_FACE_LABELS = {"sad", "fear", "angry"}   # DeepFace label space
TEXT_CONFIDENCE_THRESHOLD = 0.70
STRONG_CONFIDENCE_THRESHOLD = 0.90
MIN_DISTRESS_TEXTS = 2


def negative_scene_ratio(emotion_summary):
    """Share of stimulus scenes whose dominant facial emotion is negative."""
    if not emotion_summary:
        return 0.0
    negative = sum(1 for emotion, _count in emotion_summary.values()
                   if emotion in NEGATIVE_FACE_LABELS)
    return negative / len(emotion_summary)


def is_distress(result):
    """True if a text-emotion result counts as distress."""
    return result.label in DISTRESS_TEXT_LABELS and result.score >= TEXT_CONFIDENCE_THRESHOLD


def is_strong_distress(result):
    """True if a single text is negative with high confidence."""
    return result.label in DISTRESS_TEXT_LABELS and result.score >= STRONG_CONFIDENCE_THRESHOLD


def assess_session(user_texts, analyzer, emotion_summary):
    """Assess the whole session at its end.

    user_texts: list of strings written or spoken by the user only.
    analyzer:   object with .analyze(text) -> EmotionResult(label, score).
    emotion_summary: {scene: (dominant_emotion, readings)} from DeepFace.
    """
    texts = [t for t in user_texts if t and t.strip()]
    scored = []
    for text in texts:
        result = analyzer.analyze(text)
        scored.append({"text": text, "label": result.label, "score": result.score,
                       "distress": is_distress(result),
                       "strong": is_strong_distress(result)})

    distress_count = sum(s["distress"] for s in scored)
    strong_count = sum(s["strong"] for s in scored)
    share = (distress_count / len(scored)) if scored else 0.0
    face_ratio = negative_scene_ratio(emotion_summary)

    flag = strong_count >= 1 or distress_count >= MIN_DISTRESS_TEXTS

    reason = (f"{strong_count} text(s) negative >= {STRONG_CONFIDENCE_THRESHOLD:.2f} "
              f"(1 needed); {distress_count} of {len(scored)} texts negative >= "
              f"{TEXT_CONFIDENCE_THRESHOLD:.2f} ({MIN_DISTRESS_TEXTS} needed); "
              f"negative face in {face_ratio:.0%} of scenes "
              f"(logged only, not used for the flag)")
    return RiskAssessment(flag, reason, round(share, 3), round(face_ratio, 3), scored)
