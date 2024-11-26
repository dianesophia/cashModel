from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import pyttsx3

# text-to-speech
#engine = pyttsx3.init()
#engine.setProperty('rate', 150)
#engine.setProperty('volume', 0.9)

# Initialize webcam
cap = cv2.VideoCapture(0)  #(change to 1 if needed)
cap.set(3, 640)
cap.set(4, 480)

#YOLO model
model = YOLO("nano20.pt")

# Class names
classNames = ['Real Fifty', 'Real Five Hundred', 'Real One Hundred',
              'Real One Thousand', 'Real Twenty', 'Real Two Hundred']

prev_frame_time = 0
new_frame_time = 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture frame from webcam. Retrying...")
        continue

    new_frame_time = time.time()


    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])
            label = f'{classNames[cls]} {conf}'

            # Display label
            cvzone.putTextRect(img, label, (max(0, x1), max(35, y1)), scale=1, thickness=1)

            # Speak
            #engine.say(f"Detected {classNames[cls]}")
            #engine.runAndWait()

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
