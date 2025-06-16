# streamlit_kyc.py

import streamlit as st
import cv2
import numpy as np
import os
from deepface import DeepFace
import tempfile
import mediapipe as mp
from scipy.spatial import distance as dist
from PIL import Image

st.set_page_config(page_title="KYC Verification", layout="centered")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Constants for EAR
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
EYE_AR_THRESH = 0.25

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Face mesh detector
mp_face_mesh = mp.solutions.face_mesh
face_mesh_static = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

def eye_aspect_ratio(eye_landmarks):
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    return (A + B) / (2.0 * C)

def check_liveness(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return False
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh_static.process(img_rgb)
    if not results.multi_face_landmarks:
        return False
    landmarks = results.multi_face_landmarks[0].landmark
    h, w = img.shape[:2]

    def get_eye_coords(idxs):
        return [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in idxs]

    left_eye = get_eye_coords(LEFT_EYE_IDX)
    right_eye = get_eye_coords(RIGHT_EYE_IDX)
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    avg_ear = (left_ear + right_ear) / 2.0
    print(f"EAR: {avg_ear}")
    return avg_ear > EYE_AR_THRESH

def extract_face(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (800, 600))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    if len(faces) == 0:
        return None
    (x, y, w, h) = faces[0]
    face_img = img[y:y+h, x:x+w]
    face_path = os.path.join(UPLOAD_FOLDER, "face_" + os.path.basename(image_path))
    cv2.imwrite(face_path, face_img)
    return face_path

# ---- UI ----
st.markdown("<h1 style='text-align: center; color: #004080;'>KYC Verification</h1>", unsafe_allow_html=True)
st.markdown("---")

# Aadhaar upload
st.markdown("### 1. Upload your Aadhaar")
aadhaar_file = st.file_uploader("Upload Aadhaar (image only)", type=["jpg", "jpeg", "png"])

# Webcam input
st.markdown("### 2. Capture Live Photo")
img_data = st.camera_input("Take a live photo")

# Start KYC
st.markdown("### 3. Start KYC")
if st.button("Start KYC"):
    if aadhaar_file is None or img_data is None:
        st.error(" Please upload Aadhaar and capture your photo.")
    else:
        with st.spinner("Processing KYC..."):
            # Save Aadhaar file
            aadhaar_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            aadhaar_temp.write(aadhaar_file.getbuffer())
            aadhaar_temp_path = aadhaar_temp.name

            # Save webcam image
            webcam_image = Image.open(img_data)
            webcam_path = os.path.join(UPLOAD_FOLDER, "webcam_capture.jpg")
            webcam_image.save(webcam_path)

            # Liveness
            if not check_liveness(webcam_path):
                st.error(" Liveness check failed. Please try again.")
            else:
                # Extract faces
                aadhaar_face = extract_face(aadhaar_temp_path)
                webcam_face = extract_face(webcam_path)

                if not aadhaar_face:
                    st.error("No face detected in Aadhaar image.")
                elif not webcam_face:
                    st.error("No face detected in webcam image.")
                else:
                    try:
                        result = DeepFace.verify(img1_path=aadhaar_face, img2_path=webcam_face, enforce_detection=True)
                        if result['verified']:
                            st.success("KYC Verified!")
                            st.image(webcam_face, caption="Live Face")
                            st.image(aadhaar_face, caption="Aadhaar Face")
                        else:
                            st.error(" Face Mismatch. Verification failed.")
                    except Exception as e:
                        st.error(f"Verification error: {str(e)}")
