import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import math

# ✅ Must come right after imports
st.set_page_config(page_title="AI Gym Trainer", layout="centered")

# Load MoveNet Thunder Model
@st.cache_resource
def load_model():
    model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
    return model.signatures['serving_default']

movenet = load_model()

# Pose detection
def detect_pose(img):
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 256, 256)
    img = tf.cast(img, dtype=tf.int32)
    outputs = movenet(img)
    keypoints = outputs['output_0'].numpy()[0, 0, :, :]
    return keypoints

# Angle calculation
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# Rep counter
class PushupCounter:
    def __init__(self):
        self.count = 0
        self.stage = None

    def update(self, angle):
        if angle > 160:
            self.stage = "up"
        if angle < 90 and self.stage == "up":
            self.stage = "down"
            self.count += 1
        return self.count

counter = PushupCounter()

# Keypoint indices
LEFT_SHOULDER = 5
LEFT_ELBOW = 7
LEFT_WRIST = 9

# Webcam video processor
class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        h, w, _ = image.shape
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            keypoints = detect_pose(rgb)
            landmarks = [(int(kp[1] * w), int(kp[0] * h)) for kp in keypoints]

            if len(landmarks) >= LEFT_WRIST:
                shoulder = landmarks[LEFT_SHOULDER]
                elbow = landmarks[LEFT_ELBOW]
                wrist = landmarks[LEFT_WRIST]

                angle = calculate_angle(shoulder, elbow, wrist)
                count = counter.update(angle)

                # Draw
                cv2.putText(image, f'Angle: {int(angle)}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(image, f'Reps: {count}', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.circle(image, elbow, 8, (255, 0, 0), -1)
        except Exception as e:
            print(f"Error: {e}")

        return image

# Streamlit UI
st.title("🏋️‍♂️ AI Gym Trainer - Push-up Counter")
st.markdown("Live pose detection using MoveNet Thunder and Streamlit.")

webrtc_streamer(key="movenet", video_transformer_factory=VideoProcessor)
