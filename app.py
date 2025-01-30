import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import os
import time
import urllib.request
import bz2
from flask import Flask, render_template, Response, jsonify

# Initialize Flask app
app = Flask(__name__)

# Check if the shape predictor file exists. If not, download it.
if not os.path.isfile("shape_predictor_68_face_landmarks.dat"):
    print("Downloading shape_predictor_68_face_landmarks.dat.bz2...")
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    urllib.request.urlretrieve(url, "shape_predictor_68_face_landmarks.dat.bz2")
    print("Extracting shape_predictor_68_face_landmarks.dat.bz2...")
    with bz2.BZ2File("shape_predictor_68_face_landmarks.dat.bz2") as file:
        with open("shape_predictor_68_face_landmarks.dat", "wb") as f_out:
            f_out.write(file.read())

# Load face detection and landmark predictor models
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Constants for thresholds
EAR_THRESHOLD = 0.25  # Eye Aspect Ratio threshold
FRAME_COUNT_THRESHOLD = 4  # Number of consecutive frames with low EAR to trigger tired state
BLINK_DURATION = 1  # Minimum duration (seconds) for a blink to be counted
TIME_THRESHOLD = 3  # Time in seconds to trigger the popup
POPUP_MESSAGE = "Warning! Eye strain detected."

# Variables for blink detection and popup trigger
blink_count = 0
last_blink_time = time.time()
blinks_per_minute = 0
blink_start_time = time.time()
tired_state_detected = False
frame_count = 0
ear_below_threshold_start_time = None
popup_triggered = False
avg_ear = 0  # Global variable to store the average EAR
screen_start_time = time.time()  # Start time for screen time tracking

# Function to calculate the Eye Aspect Ratio (EAR)
def calculate_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Gamma correction to adjust brightness
def gamma_correction(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

# Preprocessing for low-light conditions
def preprocess_frame(frame):
    frame = gamma_correction(frame, gamma=1.5)  # Adjust brightness
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    gray = cv2.equalizeHist(gray)  # Enhance contrast
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply GaussianBlur to reduce noise
    return gray

def generate_frames():
    global blink_count, last_blink_time, blinks_per_minute, blink_start_time, tired_state_detected, frame_count, ear_below_threshold_start_time, popup_triggered, avg_ear
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Preprocess the frame for low-light conditions
        gray = preprocess_frame(frame)

        # Detect faces in the frame
        faces = face_detector(gray)

        for face in faces:
            landmarks = landmark_predictor(gray, face)

            # Extract eye landmarks
            left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
            right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

            # Calculate EAR
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            # Blink Detection
            if avg_ear < EAR_THRESHOLD:
                if time.time() - last_blink_time >= BLINK_DURATION:
                    blink_count += 1
                    last_blink_time = time.time()

            # Check for tiredness
            if avg_ear < EAR_THRESHOLD:
                if ear_below_threshold_start_time is None:
                    ear_below_threshold_start_time = time.time()
                elapsed_time = time.time() - ear_below_threshold_start_time

                if elapsed_time >= TIME_THRESHOLD:
                    popup_triggered = True  # Trigger popup
                    cv2.putText(frame, POPUP_MESSAGE, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                frame_count += 1
                if frame_count >= FRAME_COUNT_THRESHOLD:
                    cv2.putText(frame, "Tired (Eyes Closed)", (face.left(), face.top() - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    tired_state_detected = True
            else:
                # Reset EAR below threshold time when the user is alert
                ear_below_threshold_start_time = None
                frame_count = 0
                popup_triggered = False  # Reset popup state

                cv2.putText(frame, "Alert", (face.left(), face.top() - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Display EAR value and blinking frequency
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Draw landmarks
            for (x, y) in left_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            for (x, y) in right_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Calculate blinks per minute
        elapsed_time = time.time() - blink_start_time
        if elapsed_time >= 60:
            blinks_per_minute = blink_count
            blink_count = 0
            blink_start_time = time.time()

        # Display blinking frequency
        cv2.putText(frame, f"Blinks/min: {blinks_per_minute}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Convert the frame to a JPEG format for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame as part of the response for the video stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check_popup')
def check_popup():
    global popup_triggered
    return jsonify({"show_popup": popup_triggered})

@app.route('/status')
def status():
    global avg_ear, blinks_per_minute, tired_state_detected
    return jsonify({
        "ear": avg_ear,
        "blinksPerMinute": blinks_per_minute,
        "status": "tired" if tired_state_detected else "alert"
    })

@app.route('/screen_time')
def screen_time():
    elapsed_time = int(time.time() - screen_start_time)
    minutes = elapsed_time // 60
    return jsonify({"minutes": minutes, "seconds": elapsed_time % 60})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
