"""
test_transcripts.py
-------------------
The 10 hand-written test sentences used to benchmark the text-emotion models
(j-hartmann, tabularisai, VADER). Each entry has the text, the expected label
and a note on why the case is included. No real user data is used.
See sentiment_benchmark.py.
"""

TEST_TRANSCRIPTS = [
    {
        "text": "Honestly I feel really good today, things have been going well.",
        "expected": "happy",
        "notes": "clear positive baseline",
    },
    {
        "text": "I don't know, I've just been feeling really down lately, nothing feels exciting.",
        "expected": "sad",
        "notes": "clear negative baseline",
    },
    {
        "text": "It was a normal day, nothing much happened.",
        "expected": "neutral",
        "notes": "tests over-detection of emotion where there is none",
    },
    {
        "text": "I keep worrying about this presentation, I can't stop thinking about it.",
        "expected": "fear",
        "notes": "anxiety framed as worry rather than explicit fear language",
    },
    {
        "text": "It's just so frustrating, nobody ever listens to what I'm saying.",
        "expected": "anger",
        "notes": "frustration/anger without swearing or extreme language",
    },
    {
        "text": "I'm excited about the new job but also kind of scared I'll mess it up.",
        "expected": "mixed",
        "notes": "two emotions in one sentence - tests how each model handles ambivalence",
    },
    {
        "text": "I had this weird dream last night and I woke up feeling really unsettled.",
        "expected": "fear",
        "notes": "indirect/implicit emotional language, not a direct feeling statement",
    },
    {
        "text": "Wait, I did not expect that at all, that's wild.",
        "expected": "surprise",
        "notes": "short, informal, punctuation-light - tests robustness to casual speech",
    },
    {
        "text": "That really put me off, I couldn't stand it.",
        "expected": "disgust",
        "notes": "disgust without an explicit trigger named",
    },
    {
        "text": "I'm fine.",
        "expected": "neutral",
        "notes": "very short reply, plausibly hiding a different true state - "
                 "a known hard case for text-only sentiment",
    },
]


if __name__ == "__main__":
    for i, item in enumerate(TEST_TRANSCRIPTS, start=1):
        print(f"{i:2d}. [{item['expected']:>8}] {item['text']}")
