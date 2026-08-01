"""
plot_emotion_timeline.py
-------------------------
Plots logs/emotion_timeline.csv as a figure for the report: detected emotion
over time, with the four stimulus scenes shaded behind the readings so the
per-scene dominant emotion (and any misclassification) is visible at a glance.

This is the visual counterpart to the Stage 1 results reported in Evaluation:
the prose describes the resting-face and direction-inconsistent
misclassification findings, and this figure shows them directly.

Usage:
    pip install matplotlib
    python scripts/plot_emotion_timeline.py [path/to/emotion_timeline.csv]

Defaults to logs/emotion_timeline.csv. Saves the figure to
logs/emotion_timeline.png at 150 DPI, ready to drop into the report.
"""

import csv
import sys
from collections import Counter
from pathlib import Path

DEFAULT_CSV = "logs/emotion_timeline.csv"
OUTPUT_PNG = "logs/emotion_timeline.png"

# Fixed order so the y-axis is stable across runs regardless of which
# emotions happen to appear in a given session.
EMOTION_ORDER = ["angry", "disgust", "fear", "sad", "neutral", "surprise", "happy"]

# Muted background colours per scene, in the order scenes are presented.
SCENE_COLOURS = {
    "threatening scene": "#f2d7d5",
    "joyful scene": "#fcf3cf",
    "sad scene": "#d6eaf8",
    "relaxing scene": "#d5f5e3",
}


def load_timeline(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "time": float(row["time_seconds"]),
                "scene": row["scene"],
                "emotion": row["emotion"],
            })
    return rows


def scene_spans(rows):
    """Collapse consecutive readings into (scene, start_time, end_time) spans
    so each scene can be shaded as one block rather than per reading."""
    spans = []
    for row in rows:
        if spans and spans[-1][0] == row["scene"]:
            spans[-1][2] = row["time"]
        else:
            spans.append([row["scene"], row["time"], row["time"]])
    return spans


def dominant_per_scene(rows):
    per_scene = {}
    for row in rows:
        per_scene.setdefault(row["scene"], []).append(row["emotion"])
    return {s: Counter(e).most_common(1)[0] for s, e in per_scene.items()}


def plot(rows, output_path=OUTPUT_PNG):
    import matplotlib
    matplotlib.use("Agg")  # no display needed; write straight to file
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Only plot emotion rows actually present, but keep the canonical ordering.
    present = [e for e in EMOTION_ORDER if any(r["emotion"] == e for r in rows)]
    y_index = {emotion: i for i, emotion in enumerate(present)}

    fig, ax = plt.subplots(figsize=(11, 4.5))

    # Shade each scene block behind the readings.
    for scene, start, end in scene_spans(rows):
        ax.axvspan(start, end, color=SCENE_COLOURS.get(scene, "#eeeeee"), zorder=0)

    times = [r["time"] for r in rows]
    ys = [y_index[r["emotion"]] for r in rows]
    ax.plot(times, ys, color="#34495e", linewidth=1, alpha=0.5, zorder=2)
    ax.scatter(times, ys, color="#2c3e50", s=22, zorder=3)

    ax.set_yticks(range(len(present)))
    ax.set_yticklabels(present)
    ax.set_ylim(-0.6, len(present) - 0.4)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Detected emotion")
    ax.set_title("Detected facial emotion over one full stimulus sequence")
    ax.grid(axis="x", linestyle=":", alpha=0.4, zorder=1)

    # Legend showing which colour is which scene, in presentation order.
    seen = []
    for scene, _, _ in scene_spans(rows):
        if scene not in seen:
            seen.append(scene)
    handles = [mpatches.Patch(color=SCENE_COLOURS.get(s, "#eeeeee"), label=s) for s in seen]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=len(handles), frameon=False, fontsize=9)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV

    if not Path(csv_path).exists():
        print(f"Could not find {csv_path}")
        print("Run a full session first (python src/pipeline.py) to generate it.")
        sys.exit(1)

    rows = load_timeline(csv_path)
    if not rows:
        print(f"{csv_path} is empty.")
        sys.exit(1)

    print(f"Loaded {len(rows)} readings from {csv_path}\n")
    print("Dominant emotion per scene:")
    for scene, (emotion, count) in dominant_per_scene(rows).items():
        print(f"  {scene:<20} {emotion} ({count} readings)")
    print()

    try:
        plot(rows, OUTPUT_PNG)
    except ImportError:
        print("matplotlib not installed - run: pip install matplotlib")
        sys.exit(1)
