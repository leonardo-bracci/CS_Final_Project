import cv2
import time
import csv
from pathlib import Path
from collections import Counter

# ---- Stimulus / display config ----
CANVAS_SIZE = (960, 720)        # (width, height) of the stimulus window
PIP_SIZE = (160, 120)           # (width, height) of the webcam corner box
PIP_MARGIN = 15
PIP_CORNER = "bottom-right"
PIP_BORDER_COLOR = (255, 255, 255)  # BGR
PIP_BORDER_THICKNESS = 2
SECONDS_PER_IMAGE = 3

# True shows scene/emotion labels on screen.
# Must be False for any real session or user study: showing the participant
# which emotion a scene is meant to evoke (or what their face is currently
# reading as) biases their reaction and defeats the point of the stimulus.
DEBUG_OVERLAY = False

STIMULI_ROOT = Path(__file__).resolve().parents[1] / "samples" / "stimuli"
SCENE_LABELS = ["threatening scene", "joyful scene", "sad scene", "relaxing scene"]
SCENE_FOLDERS = {
    "threatening scene": "threatening",
    "joyful scene": "joyful",
    "sad scene": "sad",
    "relaxing scene": "relaxing",
}


def build_video_sequence(seconds_per_image=SECONDS_PER_IMAGE, stimuli_root=STIMULI_ROOT):
    """Build the (start, end, label, image_path) schedule from whatever images
    are found in samples/stimuli/<scene>/, each shown for seconds_per_image
    seconds, scenes played in SCENE_LABELS order.

    Kept as a function (not just a module-level constant) so it can be called
    with a different stimuli_root in isolation/testing without touching the
    real samples folder, and so a missing folder produces a clear warning
    rather than an import-time crash.
    """
    sequence = []
    elapsed = 0.0
    for label in SCENE_LABELS:
        folder = stimuli_root / SCENE_FOLDERS[label]
        if not folder.is_dir():
            print(f"[warn] stimulus folder not found: {folder}")
            continue
        image_paths = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        if not image_paths:
            print(f"[warn] no images found in {folder}")
            continue
        for image_path in image_paths:
            start = elapsed
            end = elapsed + seconds_per_image
            sequence.append((start, end, label, image_path))
            elapsed = end
    return sequence


# Built once at import time; get_current_scene() and get_current_stimulus()
# both default to this, but accept an explicit sequence override so tests
# don't depend on however many images happen to be on disk locally.
video_sequence = build_video_sequence()


def get_current_scene(elapsed, sequence=None):
    """Return the scene label active at the given elapsed time, or 'end'."""
    sequence = video_sequence if sequence is None else sequence
    for start, end, label, *_ in sequence:
        if start <= elapsed < end:
            return label
    return "end"


def get_current_stimulus(elapsed, sequence=None):
    """Return (label, image_path) active at the given elapsed time, or
    (None, None) once the sequence has ended."""
    sequence = video_sequence if sequence is None else sequence
    for start, end, label, image_path in sequence:
        if start <= elapsed < end:
            return label, image_path
    return None, None


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


def _pip_position(canvas_w, canvas_h, pip_w, pip_h, corner, margin):
    if corner == "top-left":
        return margin, margin
    if corner == "top-right":
        return canvas_w - pip_w - margin, margin
    if corner == "bottom-left":
        return margin, canvas_h - pip_h - margin
    return canvas_w - pip_w - margin, canvas_h - pip_h - margin  # bottom-right default


def compose_frame(stimulus_img, webcam_frame, scene_label, emotion_label="", fps=None, latency_ms=None):
    """Composite the webcam feed as a small picture-in-picture box over the
    (already canvas-sized) stimulus image, video-call style. Scene and
    emotion labels are only drawn when DEBUG_OVERLAY is True - see the
    module docstring comment on that flag for why."""
    canvas = stimulus_img.copy()
    cw, ch = CANVAS_SIZE
    pw, ph = PIP_SIZE

    pip = cv2.resize(webcam_frame, (pw, ph))
    x, y = _pip_position(cw, ch, pw, ph, PIP_CORNER, PIP_MARGIN)

    bt = PIP_BORDER_THICKNESS
    cv2.rectangle(canvas, (x - bt, y - bt), (x + pw + bt, y + ph + bt), PIP_BORDER_COLOR, -1)
    canvas[y:y + ph, x:x + pw] = pip

    if DEBUG_OVERLAY:
        cv2.putText(canvas, f"Scene: {scene_label}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        if emotion_label:
            cv2.putText(canvas, emotion_label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if fps is not None and latency_ms is not None:
            cv2.putText(canvas, f"FPS: {fps:.1f}  Latency: {latency_ms:.0f}ms",
                        (20, ch - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return canvas


def run_session():
    """Run one live webcam emotion-detection session against the real OASIS
    stimulus sequence.

    Displays each stimulus image full-size with the webcam feed composited
    as a small picture-in-picture box (like a video call), analyses emotion
    every 10th frame with DeepFace, tags each reading to the current scene,
    and returns the collected timeline. Scene and detected-emotion labels
    are not shown to the participant (see DEBUG_OVERLAY) so their reaction
    to each image stays unprompted.
    """
    from deepface import DeepFace

    if not video_sequence:
        print("No stimulus images found - check samples/stimuli/<scene>/. Aborting session.")
        return []

    cap = cv2.VideoCapture(0)
    frame_count = 0
    dominant = ""
    prev_time = time.time()
    analysis_latency = 0
    timeline = []
    session_start = time.time()

    # Cache decoded stimulus images once per path rather than re-reading from
    # disk on every frame.
    image_cache = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = time.time()
        elapsed = current_time - session_start
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        scene, image_path = get_current_stimulus(elapsed)
        if scene is None:
            break

        if image_path not in image_cache:
            img = cv2.imread(str(image_path))
            image_cache[image_path] = cv2.resize(img, CANVAS_SIZE)
        stimulus_img = image_cache[image_path]

        if frame_count % 10 == 0:
            try:
                start = time.time()
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                analysis_latency = (time.time() - start) * 1000
                dominant = result[0]['dominant_emotion']
                timeline.append((round(elapsed, 2), scene, dominant))
            except Exception as e:
                print(e)

        composed = compose_frame(
            stimulus_img, frame, scene,
            emotion_label=dominant, fps=fps, latency_ms=analysis_latency,
        )
        cv2.imshow('Emotional AI Mirror', composed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return timeline


def save_timeline_csv(timeline, path="logs/emotion_timeline.csv"):
    """Write the timeline to CSV. Kept separate so callers choose whether to save."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
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
