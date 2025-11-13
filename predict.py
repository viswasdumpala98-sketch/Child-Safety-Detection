# predict.py

import cv2
import os
from ultralytics import YOLO
from datetime import datetime

# Load model once
model = YOLO("runs/detect/child_safety_v1/weights/best.pt")
danger_items = ["battery", "candle", "dice"]

def predict_image(img_path):
    img = cv2.imread(img_path)
    results = model(img, conf=0.25)
    annotated = results[0].plot()

    detected_classes = set(model.names[int(box.cls)] for box in results[0].boxes)
    danger_found = [cls for cls in detected_classes if cls in danger_items]

    if danger_found:
        status = f"DANGER: {', '.join(danger_found)}"
        color = (0, 0, 255)
    else:
        status = "Safe"
        color = (0, 255, 0)

    cv2.putText(annotated, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    output_path = f"outputs/image_detected_{datetime.now().strftime('%H%M%S')}.jpg"
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, annotated)
    return output_path, status


def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "❌ Cannot open video"

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    os.makedirs("outputs", exist_ok=True)
    out_path = f"outputs/video_detected_{datetime.now().strftime('%H%M%S')}.mp4"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    danger_found_overall = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.25)
        annotated = results[0].plot()

        detected_classes = set(model.names[int(box.cls)] for box in results[0].boxes)
        danger_found = [cls for cls in detected_classes if cls in danger_items]
        danger_found_overall.update(danger_found)

        if danger_found:
            status = f"DANGER: {', '.join(danger_found)}"
            color = (0, 0, 255)
        else:
            status = "Safe"
            color = (0, 255, 0)

        cv2.putText(annotated, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        out.write(annotated)

    cap.release()
    out.release()

    final_status = f"DANGER: {', '.join(danger_found_overall)}" if danger_found_overall else "Safe"
    return out_path, final_status
