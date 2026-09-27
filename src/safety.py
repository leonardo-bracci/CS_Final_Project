"""
safety.py
---------
End-of-session seek-help assessment.

Replaces the earlier approach, which set seek_help by scanning the language
model's OWN reply for words such as "professional". Because the system prompt
tells the model to state it is "not a substitute for professional care",
almost every reply contained that word, and the flag fired in every logged
session (13 of 13 readable logs, including all five pilot participants).
The flag measured the model's instructions, not the user's state.

The new assessment:
  - uses only the USER's own words: the three spoken answers plus every
    chat message. The language model's replies are ignored, because they
    reflect emotions back to the user and would inflate the score;
  - reuses the existing text-emotion model (j-hartmann), so no new model
    or keyword list is introduced;
  - runs once, at the end of the session, in plain Python, so the language
    model cannot set, suppress or contradict it;
  - is explainable: the log records every message's score and the reason.

Rule: flag when EITHER
  (a) any single user text is classified as a negative emotion (sadness,
      fear, anger or disgust) with confidence >= STRONG_CONFIDENCE_THRESHOLD,
      so one clear disclosure is enough on its own; OR
  (b) at least MIN_DISTRESS_TEXTS user texts are negative with confidence
      >= TEXT_CONFIDENCE_THRESHOLD.
A count is used rather than a share: an earlier share-based rule (>= 50% of
texts) let neutral messages dilute clearly distressed ones and missed a
session containing two strongly sad messages among six texts. No keyword
list is used; detection depends on the emotion model's reading of the text,
not on intent, which is a stated limitation.

The facial data is NOT used for the flag. An earlier version also required a
negative face in at least half the stimulus scenes, but in testing a resting
face read as neutral in every scene and blocked the flag for a user whose
chat messages were clearly distressed. The negative-face ratio is still
computed and logged as supporting information only.

This is a signposting aid, not a risk or crisis assessment and not a
diagnosis. Resources are shown at the end of every session regardless.
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
