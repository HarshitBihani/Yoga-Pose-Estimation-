import streamlit as st
import numpy as np
import math, pickle
from PIL import Image
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


load_model = pickle.load(open('Model.pkl', 'rb'))

def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return round(ang + 360 if ang < 0 else ang)

def feature_list(poseLandmarks, posename):
    return [getAngle(poseLandmarks[16], poseLandmarks[14], poseLandmarks[12]),
            getAngle(poseLandmarks[14], poseLandmarks[12], poseLandmarks[24]),
            getAngle(poseLandmarks[13], poseLandmarks[11], poseLandmarks[23]),
            getAngle(poseLandmarks[15], poseLandmarks[13], poseLandmarks[11]),
            getAngle(poseLandmarks[12], poseLandmarks[24], poseLandmarks[26]),
            getAngle(poseLandmarks[11], poseLandmarks[23], poseLandmarks[25]),
            getAngle(poseLandmarks[24], poseLandmarks[26], poseLandmarks[28]),
            getAngle(poseLandmarks[23], poseLandmarks[25], poseLandmarks[27]),
            getAngle(poseLandmarks[26], poseLandmarks[28], poseLandmarks[32]),
            getAngle(poseLandmarks[25], poseLandmarks[27], poseLandmarks[31]),
            getAngle(poseLandmarks[0], poseLandmarks[12], poseLandmarks[11]),
            getAngle(poseLandmarks[0], poseLandmarks[11], poseLandmarks[12]),
            posename]


st.set_page_config(layout="wide")


st.sidebar.title('Yoga Pose Detection Project')

app_mode = st.sidebar.selectbox('Select The Pose', ['Tree', 'Mountain', 'Warrior'])

if app_mode == 'Tree':

    col1, col2 = st.columns([2, 3])
    with col1:
        with st.container():
            st.write("Tree Pose")
            image = Image.open('D:/Yoga-Pose-Detection/Tree Yoga Pose.jpg')  
            st.image(image, caption='Tree Yoga Pose')
    with col2:
        st.write("Webcam Live ")

        button = st.empty()
        start = button.button('Start')
        if start:
            stop = button.button('Stop')
            visible_message = st.empty()
            FRAME_WINDOW = st.image([])  
            accuracytxtbox = st.empty()

            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                st.write('Error: Camera is not accessible.')
            else:
                with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                    while cap.isOpened():
                        ret, frame = cap.read()

                        if not ret:
                            st.write("Error: Could not read frame from camera.")
                            break  

                        h, w, c = frame.shape

                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image.flags.writeable = False

                        results = pose.process(image)

                        image.flags.writeable = True
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                  mp_drawing.DrawingSpec(color=(240, 110, 60), thickness=2, circle_radius=2),
                                                  mp_drawing.DrawingSpec(color=(240, 60, 230), thickness=2, circle_radius=2))

                        FRAME_WINDOW.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) 
                        
                        poseLandmarks = []
                        if results.pose_landmarks:
                            for lm in results.pose_landmarks.landmark:
                                poseLandmarks.append((int(lm.x * w), int(lm.y * h)))

                        if len(poseLandmarks) == 0:
                            visible_message.text("Body Not Visible")
                            accuracytxtbox.text('')
                            continue
                        else:
                            visible_message.text("")

                            d = feature_list(poseLandmarks, 1)
                            rt_accuracy = int(round(load_model.predict(np.array(d).reshape(1, -1))[0], 0))

                            if rt_accuracy < 75:
                                accuracytxtbox.text(f"Accuracy : Not so Good {rt_accuracy}")
                            elif rt_accuracy > 75 and rt_accuracy < 85:
                                accuracytxtbox.text(f"Accuracy : Fine {rt_accuracy}")
                            elif rt_accuracy > 85 and rt_accuracy < 95:
                                accuracytxtbox.text(f"Accuracy : Very Good {rt_accuracy}")
                            elif rt_accuracy > 95 and rt_accuracy < 100:
                                accuracytxtbox.text(f"Accuracy : Near to perfection {rt_accuracy}")
                            elif rt_accuracy >= 100:
                                accuracytxtbox.text(f"Accuracy : perfect ")

                        if stop:
                            cap.release()
                            cv2.destroyAllWindows()

if app_mode == 'Mountain':

    col1, col2 = st.columns([2, 3])
    with col1:
        with st.container():
            st.write("Mountain Pose")
            image = Image.open('D:/Yoga-Pose-Detection/Mountain Yoga Pose.jpg') 
            st.image(image, caption='Mountain Yoga Pose')
    with col2:
        st.write("Webcam Live Feed")

        button = st.empty()
        start = button.button('Start')
        if start:
            stop = button.button('Stop')
            visible_message = st.empty()
            FRAME_WINDOW = st.image([])  
            accuracytxtbox = st.empty()

            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                st.write('Error: Camera is not accessible.')
            else:
                with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                    while cap.isOpened():
                        ret, frame = cap.read()

                        if not ret:
                            st.write("Error: Could not read frame from camera.")
                            break  

                        h, w, c = frame.shape

                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image.flags.writeable = False

                        results = pose.process(image)

                        image.flags.writeable = True
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                  mp_drawing.DrawingSpec(color=(240, 110, 60), thickness=2, circle_radius=2),
                                                  mp_drawing.DrawingSpec(color=(240, 60, 230), thickness=2, circle_radius=2))

                        FRAME_WINDOW.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  
                        
                        poseLandmarks = []
                        if results.pose_landmarks:
                            for lm in results.pose_landmarks.landmark:
                                poseLandmarks.append((int(lm.x * w), int(lm.y * h)))

                        if len(poseLandmarks) == 0:
                            visible_message.text("Body Not Visible")
                            accuracytxtbox.text('')
                            continue
                        else:
                            visible_message.text("")

                            d = feature_list(poseLandmarks, 2)
                            rt_accuracy = int(round(load_model.predict(np.array(d).reshape(1, -1))[0], 0))

                            if rt_accuracy < 75:
                                accuracytxtbox.text(f"Accuracy : Not so Good {rt_accuracy}")
                            elif rt_accuracy > 75 and rt_accuracy < 85:
                                accuracytxtbox.text(f"Accuracy : Fine {rt_accuracy}")
                            elif rt_accuracy > 85 and rt_accuracy < 95:
                                accuracytxtbox.text(f"Accuracy : Very Good {rt_accuracy}")
                            elif rt_accuracy > 95 and rt_accuracy < 100:
                                accuracytxtbox.text(f"Accuracy : Near to perfection {rt_accuracy}")
                            elif rt_accuracy >= 100:
                                accuracytxtbox.text(f"Accuracy : Perfect ")

                        if stop:
                            cap.release()
                            cv2.destroyAllWindows()

if app_mode == 'Warrior':

    col1, col2 = st.columns([2, 3])
    with col1:
        with st.container():
            st.write("Warrior Yoga Pose")
            image = Image.open('D:/Yoga-Pose-Detection/Warrior Yoga Pose.jpg')  
            st.image(image, caption='Warrior Yoga Pose')
    with col2:
        st.write("Webcam Live Feed")

        button = st.empty()
        start = button.button('Start')
        if start:
            stop = button.button('Stop')
            visible_message = st.empty()
            FRAME_WINDOW = st.image([])  
            accuracytxtbox = st.empty()

            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                st.write('Error: Camera is not accessible.')
            else:
                with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                    while cap.isOpened():
                        ret, frame = cap.read()

                        if not ret:
                            st.write("Error: Could not read frame from camera.")
                            break  

                        h, w, c = frame.shape

                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image.flags.writeable = False

                        results = pose.process(image)

                        image.flags.writeable = True
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                  mp_drawing.DrawingSpec(color=(240, 110, 60), thickness=2, circle_radius=2),
                                                  mp_drawing.DrawingSpec(color=(240, 60, 230), thickness=2, circle_radius=2))

                        FRAME_WINDOW.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) 
                        
                        poseLandmarks = []
                        if results.pose_landmarks:
                            for lm in results.pose_landmarks.landmark:
                                poseLandmarks.append((int(lm.x * w), int(lm.y * h)))

                        if len(poseLandmarks) == 0:
                            visible_message.text("Body Not Visible")
                            accuracytxtbox.text('')
                            continue
                        else:
                            visible_message.text("")

                            d = feature_list(poseLandmarks, 3)
                            rt_accuracy = int(round(load_model.predict(np.array(d).reshape(1, -1))[0], 0))

                            if rt_accuracy < 75:
                                accuracytxtbox.text(f"Accuracy : Not so Good {rt_accuracy}")
                            elif rt_accuracy > 75 and rt_accuracy < 85:
                                accuracytxtbox.text(f"Accuracy : Fine{rt_accuracy}")
                            elif rt_accuracy > 85 and rt_accuracy < 95:
                                accuracytxtbox.text(f"Accuracy : Very Good {rt_accuracy}")
                            elif rt_accuracy > 95 and rt_accuracy < 100:
                                accuracytxtbox.text(f"Accuracy : Near to perfection {rt_accuracy}")
                            elif rt_accuracy >= 100:
                                accuracytxtbox.text(f"Accuracy : Perfect ")

                        if stop:
                            cap.release()
                            cv2.destroyAllWindows()
