# KYC Verification System: Aadhaar Upload, Face Match, Liveness Detection using MediaPipe EAR

import cv2
import numpy as np
from deepface import DeepFace
from flask import Flask, render_template, request
import os
import base64
from werkzeug.utils import secure_filename
import mediapipe as mp
from scipy.spatial import distance as dist

app = Flask(_name_)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- MediaPipe setup for liveness check ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5)

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
EYE_AR_THRESH = 0.25  # Threshold for eyes open

def eye_aspect_ratio(eye_landmarks):
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

def check_liveness(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("[ERROR] Image not loaded for liveness check.")
        return False

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        print("[INFO] No face detected in liveness check.")
        return False

    landmarks = results.multi_face_landmarks[0].landmark
    h, w = img.shape[:2]

    def get_eye_coords(idxs):
        return [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in idxs]

    left_eye = get_eye_coords(LEFT_EYE_IDX)
    right_eye = get_eye_coords(RIGHT_EYE_IDX)

    leftEAR = eye_aspect_ratio(left_eye)
    rightEAR = eye_aspect_ratio(right_eye)
    ear = (leftEAR + rightEAR) / 2.0

    print(f"[INFO] EAR for liveness: {ear:.3f}")

    return ear > EYE_AR_THRESH


def extract_face(image_path):
    print("[INFO] Reading image for face extraction...")
    img = cv2.imread(image_path)

    if img is None:
        print("[ERROR] Image not loaded. Check path.")
        return None

    img = cv2.resize(img, (800, 600))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    print(f"[INFO] Faces found: {len(faces)}")

    if len(faces) == 0:
        return None

    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        face_path = os.path.join(UPLOAD_FOLDER, os.path.basename(image_path).split('.')[0] + '_face.jpg')
        cv2.imwrite(face_path, face_img)
        print(f"[INFO] Face saved to: {face_path}")
        return face_path

    return None

@app.route('/')
def index():
    return '''
    <h1>KYC Verification</h1>
    <form id="kycForm" action="/verify" method="post" enctype="multipart/form-data">
        Aadhaar Upload: <input type="file" name="aadhaar" required><br><br>
        <button type="button" onclick="captureAndSubmit()">Start KYC</button>
        <input type="hidden" name="webcam_image" id="webcam_image">
    </form>
    <video id="video" width="640" height="480" autoplay></video>
    <script>
    const video = document.getElementById('video');
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
        });

    function captureAndSubmit() {
        const canvas = document.createElement('canvas');
        canvas.width = 640;
        canvas.height = 480;
        const context = canvas.getContext('2d');
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataURL = canvas.toDataURL('image/jpeg');
        document.getElementById('webcam_image').value = dataURL;
        document.getElementById('kycForm').submit();
    }
    </script>
    '''

@app.route('/verify', methods=['POST'])
def verify():
    aadhaar_file = request.files['aadhaar']
    if not aadhaar_file:
        return 'No Aadhaar file uploaded.'

    aadhaar_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(aadhaar_file.filename))
    aadhaar_file.save(aadhaar_path)

    webcam_data_url = request.form.get('webcam_image')
    if not webcam_data_url:
        return 'Webcam image not received.'

    try:
        header, encoded = webcam_data_url.split(",")
        webcam_data = base64.b64decode(encoded)
        webcam_path = os.path.join(app.config['UPLOAD_FOLDER'], 'webcam_capture.jpg')
        with open(webcam_path, 'wb') as f:
            f.write(webcam_data)
    except Exception as e:
        return f"Error decoding webcam image: {str(e)}"

    if not check_liveness(webcam_path):
        return '<h2>Liveness Check Failed ❌</h2>'

    aadhaar_face_path = extract_face(aadhaar_path)
    if not aadhaar_face_path:
        return 'Face not detected in Aadhaar.'

    webcam_face_path = extract_face(webcam_path)
    if not webcam_face_path:
        return 'Face not detected in webcam image.'

    try:
        result = DeepFace.verify(img1_path=aadhaar_face_path, img2_path=webcam_face_path, enforce_detection=True)
        if result['verified']:
            return '<h2>KYC Verified ✅</h2>'
        else:
            return '<h2>Face Mismatch ❌</h2>'
    except Exception as e:
        return f"Error during face verification: {str(e)}"

if _name_ == '_main_':
    app.run(debug=True)