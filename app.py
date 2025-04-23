import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import math

# Load MoveNet Thunder model
@st.cache_resource
def load_model():
    model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
    return model.signatures['serving_default']

movenet = load_model()

def detect_pose(img):
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 256, 256)
    img = tf.cast(img, dtype=tf.int32)
    outputs = movenet(img)
    keypoints = outputs['output_0'].numpy()[0, 0, :, :]
    return keypoints

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

class ArmCounter:
    def __init__(self):
        self.count = 0
        self.stage = None

    def update(self, angle):
        if angle > 160:
            if self.stage != "up":
                self.stage = "up"
        if angle < 90 and self.stage == "up":
            self.stage = "down"
            self.count += 1
        return self.count


left_counter = ArmCounter()
right_counter = ArmCounter()

# Pose indices
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10

SKELETON_EDGES = [
    (0, 1), (1, 3), (0, 2), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 13),
    (13, 15), (12, 14), (14, 16), (11, 12)
]

class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        h, w, _ = image.shape
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            keypoints = detect_pose(rgb)

            confidence_threshold = 0.3
            landmarks = []
            for kp in keypoints:
                y, x, conf = kp
                if conf > confidence_threshold:
                    landmarks.append((int(x * w), int(y * h)))
                else:
                    landmarks.append(None)

            # Draw skeleton
            for i, j in SKELETON_EDGES:
                if landmarks[i] and landmarks[j]:
                    cv2.line(image, landmarks[i], landmarks[j], (0, 255, 255), 2)

            # LEFT ARM
                        # LEFT ARM
            if all(landmarks[k] is not None for k in [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST]):
                l_shoulder = landmarks[LEFT_SHOULDER]
                l_elbow = landmarks[LEFT_ELBOW]
                l_wrist = landmarks[LEFT_WRIST]
                left_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                _ = left_counter.update(left_angle)
                cv2.putText(image, f'L: {int(left_angle)}°', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.circle(image, l_elbow, 8, (0, 255, 0), -1)

            # RIGHT ARM
            if all(landmarks[k] is not None for k in [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST]):
                r_shoulder = landmarks[RIGHT_SHOULDER]
                r_elbow = landmarks[RIGHT_ELBOW]
                r_wrist = landmarks[RIGHT_WRIST]
                right_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
                _ = right_counter.update(right_angle)
                cv2.putText(image, f'R: {int(right_angle)}°', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.circle(image, r_elbow, 8, (255, 0, 0), -1)


            # Display rep table on video
            cv2.rectangle(image, (10, 90), (180, 150), (50, 50, 50), -1)
            cv2.putText(image, "Rep Count", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(image, f"Left Arm : {left_counter.count}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(image, f"Right Arm: {right_counter.count}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        except Exception as e:
            print("Error:", e)

        return image

# Title
st.title("🏋️ Dual Arm Pose Counter with Realtime Skeleton")

# Start webcam
webrtc_streamer(key="movenet", video_transformer_factory=VideoProcessor)
