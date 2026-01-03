import streamlit as st
import numpy as np
import cv2
import pickle
import math
import mediapipe as mp
from PIL import Image

# ---------------- LOAD MODEL ----------------
with open("Model.pkl", "rb") as f:
    model = pickle.load(f)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# ---------------- HELPER FUNCTION ----------------
def get_angle(a, b, c):
    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) -
        math.atan2(a[1] - b[1], a[0] - b[0])
    )
    return angle + 360 if angle < 0 else angle


def extract_features(landmarks, pose_id):
    return [
        get_angle(landmarks[16], landmarks[14], landmarks[12]),
        get_angle(landmarks[14], landmarks[12], landmarks[24]),
        get_angle(landmarks[13], landmarks[11], landmarks[23]),
        get_angle(landmarks[15], landmarks[13], landmarks[11]),
        get_angle(landmarks[12], landmarks[24], landmarks[26]),
        get_angle(landmarks[11], landmarks[23], landmarks[25]),
        get_angle(landmarks[24], landmarks[26], landmarks[28]),
        get_angle(landmarks[23], landmarks[25], landmarks[27]),
        get_angle(landmarks[26], landmarks[28], landmarks[32]),
        get_angle(landmarks[25], landmarks[27], landmarks[31]),
        get_angle(landmarks[0], landmarks[12], landmarks[11]),
        get_angle(landmarks[0], landmarks[11], landmarks[12]),
        pose_id
    ]


# ---------------- UI ----------------
st.set_page_config(layout="wide")
st.title("🧘 Yoga Pose Detection")

pose_name = st.sidebar.selectbox("Select Pose", ["Tree", "Mountain", "Warrior"])

POSES = {
    "Tree": ("Tree Yoga Pose.jpg", 1),
    "Mountain": ("Mountain Yoga Pose.jpg", 2),
    "Warrior": ("Warrior Yoga Pose.jpg", 3),
}

img_path, pose_id = POSES[pose_name]

col1, col2 = st.columns([2, 3])

# ---------------- LEFT COLUMN ----------------
with col1:
    st.subheader(f"{pose_name} Pose")
    img = Image.open(img_path)
    st.image(img, use_container_width=True)

# ---------------- RIGHT COLUMN ----------------
with col2:
    st.subheader("Live Camera")

    start = st.button("Start Camera")
    stop = st.button("Stop Camera")

    frame_box = st.image([])
    accuracy_box = st.empty()

    if start:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)

                if result.pose_landmarks:
                    landmarks = [
                        (int(lm.x * w), int(lm.y * h))
                        for lm in result.pose_landmarks.landmark
                    ]

                    features = extract_features(landmarks, pose_id)
                    prediction = int(model.predict(np.array(features).reshape(1, -1))[0])

                    accuracy_box.markdown(f"### Accuracy: **{prediction}%**")

                frame_box.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if stop:
                    break

        cap.release()
        cv2.destroyAllWindows()
