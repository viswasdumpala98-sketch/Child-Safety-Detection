# realtime_webcam.py

import cv2
from ultralytics import YOLO
from datetime import datetime
import os

def run_live_detection():
    print("🎥 Starting live webcam detection...")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend
    if not cap.isOpened():
        print("❌ Webcam not available or blocked.")
        return "❌ Webcam not accessible"

    model = YOLO("runs/detect/child_safety_v1/weights/best.pt")
    danger_items = ["battery", "candle", "dice"]

    os.makedirs("outputs/live", exist_ok=True)
    os.makedirs("detected_objects/live", exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame capture failed.")
            break

        results = model(frame, conf=0.25)
        annotated = results[0].plot()

        # Detection logic
        detected_classes = set(model.names[int(box.cls)] for box in results[0].boxes)
        danger_found = [cls for cls in detected_classes if cls in danger_items]

        if danger_found:
            status_msg = f"DANGER: {', '.join(danger_found)}"
            color = (0, 0, 255)
            for box in results[0].boxes:
                class_name = model.names[int(box.cls)]
                if class_name in danger_items:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = frame[y1:y2, x1:x2]
                    cv2.imwrite(f"detected_objects/live/{class_name}_{datetime.now().strftime('%H%M%S')}.jpg", crop)
        else:
            status_msg = "Safe"
            color = (0, 255, 0)

        cv2.putText(annotated, status_msg, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.imshow("Live Detection - Child Safety", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return "✅ Webcam session ended"
