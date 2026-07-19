import cv2
import time
import csv
from collections import Counter

# Simulated video sequence (start, end, label).
# Placeholder stimulus for the prototype - real calibrated content comes later.
video_sequence = [
    (0, 5, "threatening scene"),
    (5, 10, "joyful scene"),
    (10, 15, "sad scene"),
    (15, 20, "relaxing scene"),
]


def get_current_scene(elapsed):
    """Return the scene label active at the given elapsed time, or 'end'."""
    for start, end, label in video_sequence:
        if start <= elapsed < end:
            return label
    return "end"


def summarise_timeline(timeline):
    """Reduce a timeline of (time, scene, emotion) rows to the dominant emotion
    per scene. Returns a dict of {scene: (emotion, count)}."""
    scenes = {}
    for _, scene, emotion in timeline:
        scenes.setdefault(scene, []).append(emotion)
    return {
        scene: Counter(emotions).most_common(1)[0]
        for scene, emotions in scenes.items()
    }


def run_session():
    """Run one live webcam emotion-detection session.

    Captures webcam frames, analyses emotion every 10th frame with DeepFace,
    tags each reading to the current scene, and returns the collected timeline.
    """
    from deepface import DeepFace

    cap = cv2.VideoCapture(0)
    frame_count = 0
    dominant = ""
    prev_time = time.time()
    analysis_latency = 0
    timeline = []
    session_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = time.time()
        elapsed = current_time - session_start
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        scene = get_current_scene(elapsed)
        if scene == "end":
            break

        if frame_count % 10 == 0:
            try:
                start = time.time()
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                analysis_latency = (time.time() - start) * 1000
                dominant = result[0]['dominant_emotion']
                timeline.append((round(elapsed, 2), scene, dominant))
            except Exception as e:
                print(e)

        if dominant:
            cv2.putText(frame, dominant, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f"Scene: {scene}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.putText(frame, f"FPS: {fps:.1f}  Latency: {analysis_latency:.0f}ms",
                    (20, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imshow('Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return timeline


def save_timeline_csv(timeline, path="emotion_timeline.csv"):
    """Write the timeline to CSV. Kept separate so callers choose whether to save."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_seconds", "scene", "emotion"])
        writer.writerows(timeline)
    print(f"Saved to {path}")


if __name__ == "__main__":
    timeline = run_session()
    if timeline:
        save_timeline_csv(timeline)
        print("\nSession Summary:")
        for scene, (emotion, count) in summarise_timeline(timeline).items():
            print(f"{scene}: dominant emotion was {emotion} ({count} readings)")