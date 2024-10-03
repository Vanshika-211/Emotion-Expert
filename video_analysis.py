import cv2
from fer import FER
from collections import Counter

def analyze_video_emotions(video_path, max_duration=45):
    detector = FER()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_duration = 1 / fps

    emotion_counter = Counter()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed_time = (frame_count * frame_duration)

        if elapsed_time > max_duration:
            break

        emotions = detector.detect_emotions(frame)

        for emotion in emotions:
            emotion_label = emotion['emotions']
            for key in emotion_label:
                emotion_counter[key] += emotion_label[key]

        frame_count += 1

    cap.release()

    total_emotions = sum(emotion_counter.values())
    if total_emotions == 0:
        print("No emotions detected.")
        return {}

    emotion_results = {}
    for emotion, count in emotion_counter.items():
        proportion = count / total_emotions
        emotion_results[emotion] = proportion

    return emotion_results

if __name__ == "__main__":
    video_path = "path/to/your/video.mp4"
    results = analyze_video_emotions(video_path)
    print("Emotion proportions in the video:")
    for emotion, proportion in results.items():
        print(f"{emotion}: {proportion:.2f}")