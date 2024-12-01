from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import pyttsx3
import threading

# text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

# Create a separate thread for speech
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Initialize webcam
cap = cv2.VideoCapture(0)  #(change to 1 if needed)
cap.set(3, 640)
cap.set(4, 480)

#YOLO model
model = YOLO("nano40.pt")

# Class names
classNames = ['Real Fifty', 'Real Five Hundred', 'Real One Hundred',
              'Real One Thousand', 'Real Twenty', 'Real Two Hundred']

prev_frame_time = 0
new_frame_time = 0

# To avoid speaking for every detection, track last spoken label
last_spoken_label = None

# Frame skip rate (process every nth frame)
frame_skip = 3
frame_count = 0

# Track previously drawn bounding boxes to avoid duplication
previous_boxes = []

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture frame from webcam. Retrying...")
        continue

    new_frame_time = time.time()
    frame_count += 1

    # Skip frames to speed up processing
    if frame_count % frame_skip != 0:
        continue

    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Check if the box already exists in previous boxes (avoiding duplicate detections)
            is_duplicate = False
            for prev_box in previous_boxes:
                prev_x1, prev_y1, prev_x2, prev_y2 = prev_box
                if (abs(x1 - prev_x1) < 10 and abs(y1 - prev_y1) < 10 and
                        abs(x2 - prev_x2) < 10 and abs(y2 - prev_y2) < 10):
                    is_duplicate = True
                    break

            if not is_duplicate:
                # Draw bounding box if it's not a duplicate
                cvzone.cornerRect(img, (x1, y1, w, h))

                # Add this box to the previous boxes list
                previous_boxes.append((x1, y1, x2, y2))

                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100

                # Class Name
                cls = int(box.cls[0])
                label = f'{classNames[cls]} {conf}'

                # Display label
                cvzone.putTextRect(img, label, (max(0, x1), max(35, y1)), scale=1, thickness=1)

                # Speak only when new object is detected
                if classNames[cls] != last_spoken_label:
                    # Use a separate thread to speak, so it doesn't block the main thread
                    threading.Thread(target=speak, args=(f"Detected {classNames[cls]}",)).start()
                    last_spoken_label = classNames[cls]

    # Calculate FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"FPS: {fps:.2f}")

    # Display the frame
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
