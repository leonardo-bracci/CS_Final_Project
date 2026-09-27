"""
eval_seek_help.py
-----------------
Re-scores every saved session log with the redesigned seek-help rule
(src/safety.py) and compares it with the flag the original method logged.

The original method set seek_help by scanning the language model's own reply
for words like "professional", and fired in every session. The new rule uses
only the user's own words (three spoken answers + chat messages), scored by
the j-hartmann text-emotion model. Because every log already stores those
texts, the pilot participants' sessions can be re-assessed without new
sessions: same people, same words, old rule vs new rule.

Pilot sessions are the five non-demo runs from 7 August 2026.

Usage (from the project root):
    python scripts/eval_seek_help.py

Output: a table in the terminal and logs/seek_help_eval.csv.
"""

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from safety import assess_session          # noqa: E402
from sentiment import SentimentAnalyzer    # noqa: E402

LOGS_DIR = PROJECT_ROOT / "logs"
OUT_PATH = LOGS_DIR / "seek_help_eval.csv"
PILOT_DATE = "2026-08-07"


def load_session(path):
    """Return the parsed log, or None if the file is not valid JSON."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def user_texts(log):
    """The user's own words only: spoken answers + chat messages. The first
    'user' entry in chat_history is the structured profile prompt built by
    the pipeline, not something the user typed, so it is skipped."""
    answers = [pair.get("answer", "") for pair in log.get("spoken_qa", [])]
    user_msgs = [m["content"] for m in log.get("chat_history", []) if m.get("role") == "user"]
    return answers + user_msgs[1:]


def face_summary(log):
    """Rebuild {scene: (emotion, readings)} from the logged summary."""
    return {scene: (v["emotion"], v["readings"])
            for scene, v in log.get("emotion_summary", {}).items()}


def main():
    analyzer = SentimentAnalyzer()
    rows, skipped = [], []

    for path in sorted(LOGS_DIR.glob("session_*.json")):
        log = load_session(path)
        if log is None:
            skipped.append(path.name)
            continue

        risk = assess_session(user_texts(log), analyzer, face_summary(log))
        timestamp = log.get("timestamp", path.stem)
        is_pilot = timestamp.startswith(PILOT_DATE) and not log.get("demo_mode")
        old_flag = log.get("profile", {}).get("seek_help")
        # Logs written after the redesign already hold the new flag, so the
        # "old" column is only meaningful for sessions logged before it.
        redesigned = "seek_help_assessment" in log

        rows.append({
            "session": timestamp[:19],
            "pilot": "yes" if is_pilot else "no",
            "demo_mode": bool(log.get("demo_mode")),
            "old_flag": "n/a (new method)" if redesigned else old_flag,
            "new_flag": risk.flag,
            "texts": len(risk.scored_texts),
            "negative_texts": sum(s["distress"] for s in risk.scored_texts),
            "strong_negative_texts": sum(s["strong"] for s in risk.scored_texts),
            "reason": risk.reason,
        })

    LOGS_DIR.mkdir(exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'session':<20} {'pilot':<6} {'old':<18} {'new':<6} {'neg':>4} {'strong':>7} {'texts':>6}")
    for r in rows:
        print(f"{r['session']:<20} {r['pilot']:<6} {str(r['old_flag']):<18} "
              f"{str(r['new_flag']):<6} {r['negative_texts']:>4} "
              f"{r['strong_negative_texts']:>7} {r['texts']:>6}")

    pilots = [r for r in rows if r["pilot"] == "yes"]
    if pilots:
        old_n = sum(r["old_flag"] is True for r in pilots)
        new_n = sum(r["new_flag"] is True for r in pilots)
        print(f"\nPilot sessions: old method flagged {old_n}/{len(pilots)}, "
              f"new method flagged {new_n}/{len(pilots)}")
    if skipped:
        print(f"Skipped unreadable logs: {', '.join(skipped)}")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
