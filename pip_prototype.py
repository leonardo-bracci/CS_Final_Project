"""
pip_prototype.py
-----------------
Throwaway prototype to test the picture-in-picture layout (stimulus image
big in the middle, webcam feed small in the corner, like a video call)
before wiring it into emotion_detector.py for real.

Run this directly, look at the window, tweak the CONFIG values below until
it feels right, then tell me the final numbers and I'll fold them into
run_session().

Usage:
    python pip_prototype.py

Press 'q' to quit. Cycles through whatever images it finds in
samples/stimuli/<scene>/, 3 seconds each, in folder order.
"""

import time
from pathlib import Path

import cv2

# ---- CONFIG: tweak these and re-run ----
CANVAS_SIZE = (960, 720)       # (width, height) of the main window
PIP_SIZE = (160, 120)          # (width, height) of the webcam corner box
PIP_MARGIN = 15                # px from the edges
PIP_CORNER = "bottom-right"    # one of: top-left, top-right, bottom-left, bottom-right
PIP_BORDER_COLOR = (255, 255, 255)  # BGR
PIP_BORDER_THICKNESS = 2
SECONDS_PER_IMAGE = 3
STIMULI_ROOT = Path(__file__).resolve().parent / "samples" / "stimuli"
SCENES = ["threatening scene", "joyful scene", "sad scene", "relaxing scene"]
# folder names under samples/stimuli/ - adjust if yours differ, e.g. "threatening"
SCENE_FOLDERS = {
    "threatening scene": "threatening",
    "joyful scene": "joyful",
    "sad scene": "sad",
    "relaxing scene": "relaxing",
}
# -----------------------------------------


def load_scene_images():
    """Load every image found in each scene folder, resized once to canvas size."""
    scenes = []
    for label in SCENES:
        folder = STIMULI_ROOT / SCENE_FOLDERS[label]
        paths = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.jpeg")) + sorted(folder.glob("*.png"))
        if not paths:
            print(f"[warn] no images found in {folder}")
            continue
        images = []
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                print(f"[warn] could not read {p}")
                continue
            img = cv2.resize(img, CANVAS_SIZE)
            images.append(img)
        scenes.append((label, images))
    return scenes


def pip_position(canvas_w, canvas_h, pip_w, pip_h, corner, margin):
    if corner == "top-left":
        return margin, margin
    if corner == "top-right":
        return canvas_w - pip_w - margin, margin
    if corner == "bottom-left":
        return margin, canvas_h - pip_h - margin
    # default: bottom-right
    return canvas_w - pip_w - margin, canvas_h - pip_h - margin


def compose_frame(stimulus_img, webcam_frame, scene_label, emotion_label=""):
    canvas = stimulus_img.copy()
    cw, ch = CANVAS_SIZE
    pw, ph = PIP_SIZE

    pip = cv2.resize(webcam_frame, (pw, ph))
    x, y = pip_position(cw, ch, pw, ph, PIP_CORNER, PIP_MARGIN)

    # border: draw a slightly larger filled rect behind the pip, then paste the pip on top
    bt = PIP_BORDER_THICKNESS
    cv2.rectangle(canvas, (x - bt, y - bt), (x + pw + bt, y + ph + bt), PIP_BORDER_COLOR, -1)
    canvas[y:y + ph, x:x + pw] = pip

    # scene label near the top of the main image
    cv2.putText(canvas, f"Scene: {scene_label}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    # emotion label just above the pip box, so it's visually tied to "you"
    if emotion_label:
        cv2.putText(canvas, emotion_label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return canvas


def run_prototype():
    scenes = load_scene_images()
    if not scenes:
        print("No stimulus images found - check STIMULI_ROOT and SCENE_FOLDERS at the top of this file.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    for label, images in scenes:
        for img in images:
            start = time.time()
            while time.time() - start < SECONDS_PER_IMAGE:
                ret, frame = cap.read()
                if not ret:
                    break
                composed = compose_frame(img, frame, label, emotion_label="(prototype)")
                cv2.imshow("Stimulus prototype (press q to quit)", composed)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

    cap.release()
    cv2.destroyAllWindows()
    print("Prototype sequence finished.")


if __name__ == "__main__":
    run_prototype()
