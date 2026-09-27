"""
emotion_benchmark.py
--------------------
Compares DeepFace (used in emotion_detector.py) with FER, a separate
facial-emotion library with its own model and face detector, on the same
live webcam frames. It records per-frame latency and whether the two models
agree, and checks how each behaves on a resting, neutral face.

Usage (from the project root):
    python scripts/emotion_benchmark.py [duration_seconds]

Samples one frame per second for duration_seconds (default 15), prints a
side-by-side comparison and saves it as emotion_benchmark_results.csv.
Keep a neutral face for the first few seconds of the run.
"""

import csv
import sys
import time
from pathlib import Path

import cv2


def analyze_deepface(frame):
    from deepface import DeepFace
    start = time.perf_counter()
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        label = result[0]['dominant_emotion']
        score = result[0]['emotion'][label]
    except Exception as e:
        label, score = f"error: {e}", None
    latency_ms = (time.perf_counter() - start) * 1000
    return label, score, latency_ms


def analyze_fer(detector, frame):
    start = time.perf_counter()
    try:
        result = detector.detect_emotions(frame)
        if not result:
            label, score = "no face detected", None
        else:
            emotions = result[0]["emotions"]
            label = max(emotions, key=emotions.get)
            score = emotions[label]
    except Exception as e:
        label, score = f"error: {e}", None
    latency_ms = (time.perf_counter() - start) * 1000
    return label, score, latency_ms


def run_benchmark(duration_seconds=15):
    from fer.fer import FER
    fer_detector = FER()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return []

    print(f"Running for {duration_seconds}s, sampling ~1 frame/sec.")
    print("Sit still with a neutral/resting face for the first several seconds.\n")

    results = []
    start_time = time.time()
    last_sample = 0

    while time.time() - start_time < duration_seconds:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed = time.time() - start_time
        if elapsed - last_sample < 1.0:
            continue
        last_sample = elapsed

        df_label, df_score, df_latency = analyze_deepface(frame)
        fer_label, fer_score, fer_latency = analyze_fer(fer_detector, frame)

        row = {
            "elapsed_s": round(elapsed, 1),
            "deepface_label": df_label,
            "deepface_score": round(df_score, 3) if df_score is not None else None,
            "deepface_latency_ms": round(df_latency, 1),
            "fer_label": fer_label,
            "fer_score": round(fer_score, 3) if fer_score is not None else None,
            "fer_latency_ms": round(fer_latency, 1),
        }
        results.append(row)
        print(f"[{row['elapsed_s']:>5.1f}s] DeepFace: {df_label:<10} ({df_latency:>6.1f}ms)   "
              f"FER: {fer_label:<10} ({fer_latency:>6.1f}ms)")

    cap.release()
    return results


def save_csv(results, path="logs/emotion_benchmark_results.csv"):
    if not results:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    results = run_benchmark(duration)

    if results:
        save_csv(results)

        avg_df_latency = sum(r["deepface_latency_ms"] for r in results) / len(results)
        avg_fer_latency = sum(r["fer_latency_ms"] for r in results) / len(results)
        print(f"\nAverage latency - DeepFace: {avg_df_latency:.1f}ms, FER: {avg_fer_latency:.1f}ms")

        print("\nLabel counts:")
        for model in ("deepface_label", "fer_label"):
            from collections import Counter
            counts = Counter(r[model] for r in results)
            print(f"  {model}: {dict(counts)}")
