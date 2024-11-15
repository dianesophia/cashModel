
import cv2
import cvzone
import math
import time
from ultralytics import YOLO
import threading
import pyttsx3

# Initialize the TTS engine
engine = pyttsx3.init()

# Class names for object detection
'''
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]
'''
#nc: 6

classNames = ['Real Fifty', 'Fifty', 'Real Five Hundred', 'Real One Hundred', 'Real One Thousand', 'Real Twenty', 'Real Two Hundred']

# Load the YOLO model
def load_model():
    #model = YOLO("../Yolo-Weights/yolov8n.pt")
    model = YOLO("moneyModel.pt")
    return model

# Function to generate and play TTS sound based on the class name
def play_tts(text):
    engine.say(text)
    engine.runAndWait()

# Function to process a single frame and return detection results
def process_frame(model, img, output_interval, last_output_time):
    results = model(img, stream=True, conf=0.5)
    current_detections = set()
    current_time = time.time()

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            cvzone.cornerRect(img, (x1, y1, w, h))
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = classNames[cls]
            current_detections.add(class_name)
            cvzone.putTextRect(img, f'{class_name} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    for obj in current_detections:
        if obj not in last_output_time or current_time - last_output_time[obj] >= output_interval:
            threading.Thread(target=play_tts, args=(obj,)).start()
            last_output_time[obj] = current_time

    return img, last_output_time
