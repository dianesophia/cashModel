# app.py
import streamlit as st
import cv2
import time
from model1 import load_model, process_frame

# Initialize YOLO model with caching in Streamlit
@st.cache_resource
def get_yolo_model():
    return load_model()

model = get_yolo_model()

st.title("Real-Time Object Detection with Audio Alerts")
st.write("Click 'Start Detection' to begin.")

# Button to start detection
if st.button("Start Detection"):
    cap = cv2.VideoCapture(0)
    frame_count = 0
    frame_skip = 2
    output_interval = 10  # Time interval in seconds for audio output
    last_output_time = {}

    # Streamlit video display
    stframe = st.empty()

    while True:
        success, img = cap.read()
        if not success:
            st.write("Failed to grab frame.")
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Process frame and retrieve updated last_output_time
        img, last_output_time = process_frame(model, img, output_interval, last_output_time)

        # Convert BGR to RGB for Streamlit display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        stframe.image(img_rgb, channels="RGB")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
