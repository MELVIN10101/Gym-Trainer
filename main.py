# ✅ AI Gym Trainer WebApp using Streamlit + MoveNet Thunder + OpenCV + Background Removal

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import math
import mediapipe as mp

# Load MoveNet Thunder model once
@st.cache_resource
def load_model():
    model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
    return model.signatures['serving_default']

movenet = load_model()

# Function to run pose detection on image
def detect_pose(img):
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 256, 256)
    img = tf.cast(img, dtype=tf.int32)
    outputs = movenet(img)
    keypoints = outputs['output_0'].numpy()[:, 0, :, :]
    return keypoints[0]

# Function to calculate angle between 3 points
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# Keypoint indices
LEFT_SHOULDER = 5
LEFT_ELBOW = 7
LEFT_WRIST = 9

# Rep counter
class PushupCounter:
    def __init__(self):
        self.count = 0
        self.stage = None

    def update(self, angle):
        if angle > 160:
            self.stage = "up"
        if angle < 90 and self.stage == 'up':
            self.stage = "down"
            self.count += 1
        return self.count

counter = PushupCounter()

# Setup Mediapipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentor = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Video processing class
class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        # Apply selfie segmentation
        results = segmentor.process(rgb)
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.3
        background = np.zeros(image.shape, dtype=np.uint8)
        person_only = np.where(condition, image, background)

        try:
            keypoints = detect_pose(rgb)
            landmarks = [(int(kp[1]*w), int(kp[0]*h)) for kp in keypoints]

            if len(landmarks) > LEFT_WRIST:
                shoulder = landmarks[LEFT_SHOULDER]
                elbow = landmarks[LEFT_ELBOW]
                wrist = landmarks[LEFT_WRIST]

                angle = calculate_angle(shoulder, elbow, wrist)
                count = counter.update(angle)

                cv2.putText(person_only, f"Angle: {int(angle)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(person_only, f"Reps: {count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.circle(person_only, elbow, 8, (255, 0, 0), -1)

        except Exception as e:
            st.error(f"Error in pose detection: {e}")

        return person_only

# Streamlit App UI
st.set_page_config(page_title="Gym Trainer AI", layout="centered")
st.title("🏋️ AI Gym Trainer - Push-up Counter")
st.markdown("Real-time push-up counter using MoveNet and your webcam with background removal.")

webrtc_streamer(key="pose-stream", video_transformer_factory=VideoProcessor)
