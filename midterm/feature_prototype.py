import cv2
import time
import csv
import matplotlib.pyplot as plt
import ollama
from deepface import DeepFace

# Simulated video sequence (start, end, label)
video_sequence = [
    (0, 5, "threatening scene"),
    (5, 10, "joyful scene"),
    (10, 15, "sad scene"),
    (15, 20, "relaxing scene"),
]

def get_current_scene(elapsed):
    for start, end, label in video_sequence:
        if start <= elapsed < end:
            return label
    return "end"

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

    cv2.putText(frame, f"FPS: {fps:.1f}  Latency: {analysis_latency:.0f}ms", (20, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if timeline:
    # Save CSV
    with open("emotion_timeline.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_seconds", "scene", "emotion"])
        for t, s, e in timeline:
            writer.writerow([t, s, e])
    print("Saved to emotion_timeline.csv")

    # Print summary
    print("\nSession Summary:")
    from collections import Counter
    scenes = {}
    for _, scene, emotion in timeline:
        if scene not in scenes:
            scenes[scene] = []
        scenes[scene].append(emotion)
    
    for scene, emotions in scenes.items():
        most_common = Counter(emotions).most_common(1)[0]
        print(f"{scene}: dominant emotion was {most_common[0]} ({most_common[1]} readings)")



# Build prompt from session summary
prompt = "You are a psychological profiling assistant. Based on the following emotional responses recorded during a video stimulus, provide a brief initial psychological observation.\n\n"

for scene, emotions in scenes.items():
    most_common = Counter(emotions).most_common(1)[0]
    prompt += f"During the '{scene}', the dominant emotion was {most_common[0]}.\n"

prompt += "\nProvide a short, empathetic observation about what these responses might suggest."

print("\nSending to Ollama...")
response = ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': prompt}])
print("\nOllama response:")
print(response['message']['content'])
