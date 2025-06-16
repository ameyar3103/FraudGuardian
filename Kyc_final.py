import cv2
import numpy as np
from deepface import DeepFace
from flask import Flask, request, Response
import os
import base64
from werkzeug.utils import secure_filename
import mediapipe as mp
from scipy.spatial import distance as dist

app = Flask(_name_)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Setup MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh_static = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# EAR constants
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
EYE_AR_THRESH = 0.25

def eye_aspect_ratio(eye_landmarks):
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    return (A + B) / (2.0 * C)

def check_liveness(image_path):
    img = cv2.imread(image_path)
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

    print(f"[INFO] EAR for liveness: {avg_ear:.3f}")
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

    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        face_path = os.path.join(UPLOAD_FOLDER, os.path.basename(image_path).split('.')[0] + '_face.jpg')
        cv2.imwrite(face_path, face_img)
        return face_path
    return None

def gen_frames():
    cap = cv2.VideoCapture(0)
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as live_mesh:

        drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 0), thickness=1, circle_radius=0)

        while True:
            success, frame = cap.read()
            if not success:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = live_mesh.process(rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=drawing_spec
                    )

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>KYC Verification</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f0f2f5;
                text-align: center;
                padding: 30px;
            }
            form {
                background-color: white;
                padding: 20px;
                border-radius: 12px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                display: inline-block;
            }
            input[type="file"], button {
                margin: 10px;
                padding: 10px;
                font-size: 16px;
            }
            button {
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 6px;
                cursor: pointer;
            }
            button:hover {
                background-color: #0056b3;
            }
            #video-preview {
                margin-top: 20px;
                border: 2px solid #ccc;
                border-radius: 8px;
            }
        </style>
    </head>
    <body>
        <h1>KYC Verification</h1>
        <form id="kycForm" action="/verify" method="post" enctype="multipart/form-data">
            <input type="file" name="aadhaar" required><br>
            <input type="hidden" name="webcam_image" id="webcam_image">
            <button type="button" onclick="captureAndSubmit()">Start Verification</button>
        </form>
        <div>
            <h3>Live Webcam Feed</h3>
            <img id="video-preview" src="/video_feed" width="640" height="480">
        </div>
        <script>
        function captureAndSubmit() {
            const video = document.getElementById('video-preview');
            const canvas = document.createElement('canvas');
            canvas.width = 640;
            canvas.height = 480;
            const context = canvas.getContext('2d');
            const img = new Image();
            img.crossOrigin = "Anonymous";
            img.src = video.src;
            img.onload = () => {
                context.drawImage(img, 0, 0, canvas.width, canvas.height);
                const dataURL = canvas.toDataURL('image/jpeg');
                document.getElementById('webcam_image').value = dataURL;
                document.getElementById('kycForm').submit();
            };
        }
        </script>
    </body>
    </html>
    '''

def render_result_page(title, message, status="success"):
    color = "#28a745" if status == "success" else "#dc3545"
    # emoji = "✅" if status == "success" else "❌"
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f8f9fa;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }}
            .result-card {{
                background-color: white;
                padding: 40px;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                text-align: center;
                width: 90%;
                max-width: 400px;
            }}
            .result-card h1 {{
                font-size: 2.5em;
                color: {color};
                margin-bottom: 10px;
            }}
            .result-card p {{
                font-size: 1.2em;
                color: #555;
            }}
            .emoji {{
                font-size: 4em;
                margin-bottom: 20px;
            }}
            .btn {{
                margin-top: 20px;
                padding: 10px 20px;
                background-color: #007bff;
                color: white;
                text-decoration: none;
                border-radius: 6px;
            }}
            .btn:hover {{
                background-color: #0056b3;
            }}
        </style>
    </head>
    <body>
        <div class="result-card">
            <div class="emoji">{emoji}</div>
            <h1>{title}</h1>
            <p>{message}</p>
            <a href="/" class="btn">Try Again</a>
        </div>
    </body>
    </html>
    '''

@app.route('/verify', methods=['POST'])
def verify():
    aadhaar_file = request.files['aadhaar']
    if not aadhaar_file:
        return render_result_page("Upload Error", "No Aadhaar file uploaded.", status="error")

    aadhaar_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(aadhaar_file.filename))
    aadhaar_file.save(aadhaar_path)

    webcam_data_url = request.form.get('webcam_image')
    if not webcam_data_url:
        return render_result_page("Capture Error", "Webcam image not received.", status="error")

    try:
        header, encoded = webcam_data_url.split(",")
        webcam_data = base64.b64decode(encoded)
        webcam_path = os.path.join(app.config['UPLOAD_FOLDER'], 'webcam_capture.jpg')
        with open(webcam_path, 'wb') as f:
            f.write(webcam_data)
    except Exception as e:
        return render_result_page("Image Error", f"Failed to decode webcam image: {str(e)}", status="error")

    if not check_liveness(webcam_path):
        return render_result_page("Liveness Check Failed", "Please ensure your eyes are open and well-lit.", status="error")

    aadhaar_face_path = extract_face(aadhaar_path)
    if not aadhaar_face_path:
        return render_result_page("Face Detection Failed", "No face detected in Aadhaar image.", status="error")

    webcam_face_path = extract_face(webcam_path)
    if not webcam_face_path:
        return render_result_page("Face Detection Failed", "No face detected in webcam image.", status="error")

    try:
        result = DeepFace.verify(img1_path=aadhaar_face_path, img2_path=webcam_face_path, enforce_detection=True)
        if result['verified']:
            return render_result_page("KYC Verified", "Your identity has been successfully verified.", status="success")
        else:
            return render_result_page("Face Mismatch", "The faces do not match. Please try again.", status="error")
    except Exception as e:
        return render_result_page("Verification Error", f"Error during face verification: {str(e)}", status="error")

if _name_ == '_main_':
    app.run(debug=True)