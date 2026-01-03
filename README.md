# 🧘 Yoga Pose Detection – Real-Time Accuracy App
<img width="1911" height="844" alt="Screenshot 2026-01-03 163138" src="https://github.com/user-attachments/assets/033cb15c-5d6d-464b-b4a7-8fed22f14527" />

This project is a **Real Time yoga pose Detection web application** that uses a **live webcam feed** to analyze body posture and calculate **pose accuracy** using a trained Machine Learning model.

The application is built with **Streamlit** for the UI and **MediaPipe** for pose landmark detection.

---

## 📌 Project Overview

- Detects human body landmarks in real time
- Calculates joint angles from landmarks
- Predicts **pose accuracy (%)**
- Designed for **learning, demos, and academic projects**
- Runs directly in the browser

---

## 🚀 Key Features

- ✅ Live webcam pose detection  
- ✅ Real time accuracy calculation  
- ✅ Multiple yoga pose support  
- ✅ Clean UI with Streamlit  



---

## 🧘 Supported Yoga Poses

- **Tree Pose**
- **Mountain Pose**
- **Warrior Pose**

(Each pose uses a different trained pose label for accuracy calculation.)

---

## 🛠️ Technologies Used

- **Python 3.10**
- **Streamlit** – Web UI
- **MediaPipe** – Pose landmark detection
- **OpenCV** – Webcam handling
- **scikit-learn** – Machine learning model
- **NumPy** – Numerical operations
- **Pillow (PIL)** – Image handling

---
## Usage

To use the Yoga Pose Detection application:

- **Download or Clone:**  
  Get the project files by either downloading the repository as a ZIP file or cloning it using Git:
  git clone https://github.com/HarshitBihani/Yoga-Pose-Estimation.git
  
- **Install Dependencies:**
  Navigate to the project directory and install the required Python libraries:

  bash
  Copy code
  pip install -r requirements.txt
  Run the Application:
  Start the Streamlit application using the following command:

  bash
  Copy code
  streamlit run Main.py

- **Open in Browser:**
The application will automatically open in your default web browser.

- **Select a Yoga Pose:**
Use the sidebar menu to select a yoga pose (Tree, Mountain, or Warrior).
A reference image of the selected pose will be displayed.

- **Start Pose Detection:**
Click the Start Camera button to begin live pose analysis.

- **View Accuracy:**
Perform the selected yoga pose in front of the camera.
The system will analyze your posture and display the pose accuracy (%) in real time.

- **Stop the Camera:**
Click the Stop Camera button to end the session.

