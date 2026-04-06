"""
EC2-2 — Face Recognition Server
Flask server that detects and recognizes faces using the face_recognition library.
Compares detected faces against a known-faces database loaded at startup.
Runs on port 5002.
"""

from flask import Flask, request, jsonify
import face_recognition
import cv2
import numpy as np
import os
import re
import boto3
from datetime import datetime

app = Flask(__name__)

s3 = boto3.client('s3')
BUCKET_NAME = "surveillance-frames"

# ─── Known Faces Database (loaded at startup) ───
known_face_encodings = []
known_face_names = []

# ─── Configuration ───
KNOWN_FACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "known_faces")
TOLERANCE = 0.6          # Lower = stricter matching (default 0.6)
DETECTION_MODEL = "hog"  # "hog" for CPU (fast), "cnn" for GPU (accurate)


def load_known_faces():
    """
    Load and encode all known face images from the known_faces/ directory.
    Each image file should be named as: person_name.jpg (or .png, .jpeg)
    The filename (without extension) becomes the person's label.
    
    Example structure:
        known_faces/
        ├── devashish.jpg
        ├── john_doe.jpg
        └── alice.png
    """
    global known_face_encodings, known_face_names

    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        print(f"📁 Created empty known_faces/ directory at: {KNOWN_FACES_DIR}")

    # Phase 11: S3 Sync
    try:
        print(f"\n☁️ Syncing known faces from s3://{BUCKET_NAME}/known_faces/")
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix="known_faces/")
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                if key.endswith('/'): continue
                
                local_path = os.path.join(KNOWN_FACES_DIR, os.path.basename(key))
                if not os.path.exists(local_path):
                    s3.download_file(BUCKET_NAME, key, local_path)
                    print(f"   ⬇️ Downloaded: {os.path.basename(key)}")
    except Exception as e:
        print(f"   ⚠️ S3 Sync skipped: {e}")

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    loaded = 0

    for filename in os.listdir(KNOWN_FACES_DIR):
        name, ext = os.path.splitext(filename)
        if ext.lower() not in valid_extensions:
            continue

        filepath = os.path.join(KNOWN_FACES_DIR, filename)

        try:
            # Load image and compute face encoding
            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])
                
                # Phase 11: Multi-Angle name regex (e.g., devashish_1 -> devashish)
                clean_name = name
                match = re.match(r"^(.+?)[_-](?:\d+|front|side|left|right|up|down)$", name, re.IGNORECASE)
                if match:
                    clean_name = match.group(1)
                
                clean_name = clean_name.replace("_", " ").title()
                known_face_names.append(clean_name)
                loaded += 1
                print(f"   ✅ Loaded: {clean_name} ({filename})")
            else:
                print(f"   ⚠️  No face found in: {filename} — skipping")

        except Exception as e:
            print(f"   ❌ Error loading {filename}: {e}")

    print(f"\n   📊 Total known faces loaded: {loaded}")


def check_liveness(face_image):
    """ Phase 11: Anti-Spoofing Laplacian Variance Heuristic """
    gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    # A flat photo/screen often lacks micro-textures (variance < ~60)
    return variance > 60


def recognize_faces(frame_bytes):
    """
    Detect and recognize all faces in a frame.
    
    Steps:
        1. Decode JPEG bytes to numpy array
        2. Detect face locations using HOG detector
        3. Compute 128-d face encodings for each detected face
        4. Compare each encoding against known-faces DB
        5. Return list of faces with labels and confidence
    
    Returns:
        dict: {faces_detected, faces, unknown_count, known_count, threat}
    """
    # Decode JPEG bytes to numpy array (RGB for face_recognition)
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame_bgr is None:
        return {
            "faces_detected": 0,
            "faces": [],
            "unknown_count": 0,
            "known_count": 0,
            "threat": False,
            "message": "Failed to decode frame"
        }

    # Convert BGR (OpenCV) to RGB (face_recognition expects RGB)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Resize frame for faster processing (scale down by 50%)
    small_frame = cv2.resize(frame_rgb, (0, 0), fx=0.5, fy=0.5)

    # Detect face locations
    face_locations = face_recognition.face_locations(small_frame, model=DETECTION_MODEL)

    # If no faces found
    if len(face_locations) == 0:
        return {
            "faces_detected": 0,
            "faces": [],
            "unknown_count": 0,
            "known_count": 0,
            "threat": False,
            "message": "No faces detected in frame"
        }

    # Compute 128-d encodings for each detected face
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    faces_result = []
    unknown_count = 0
    known_count = 0

    for i, face_encoding in enumerate(face_encodings):
        name = "Unknown"
        confidence = 0.0

        if len(known_face_encodings) > 0:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            best_distance = face_distances[best_match_index]
            confidence = round(1.0 - best_distance, 4)

            if best_distance <= TOLERANCE:
                name = known_face_names[best_match_index]
                known_count += 1
            else:
                unknown_count += 1
        else:
            unknown_count += 1
            confidence = 0.0

        # Extract Face Crop
        top_raw, right_raw, bottom_raw, left_raw = face_locations[i]
        face_crop = small_frame[top_raw:bottom_raw, left_raw:right_raw]

        # Phase 11: Liveness Check & Auto-Upload Queue
        if face_crop.size > 0:
            is_live = check_liveness(face_crop)
            if not is_live:
                name = "SPOOF_DETECTED"
                confidence = 0.0
                if name != "Unknown": # Override counts
                    known_count = max(0, known_count - 1)
                    unknown_count += 1
            
            # If Unknown (or spoof), Auto-Upload to S3 Pending Queue
            elif name == "Unknown":
                try:
                    face_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
                    _, buffer = cv2.imencode('.jpg', face_bgr)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    s3.put_object(Bucket=BUCKET_NAME, Key=f"pending_faces/unknown_{ts}.jpg", Body=buffer.tobytes(), ContentType="image/jpeg")
                except: pass

        location = {
            "top": top_raw * 2,
            "right": right_raw * 2,
            "bottom": bottom_raw * 2,
            "left": left_raw * 2
        }

        faces_result.append({
            "name": name,
            "confidence": confidence,
            "location": location
        })

    # Threat = any unknown face detected
    threat = unknown_count > 0

    return {
        "faces_detected": len(face_locations),
        "faces": faces_result,
        "unknown_count": unknown_count,
        "known_count": known_count,
        "threat": threat,
        "message": f"Detected {len(face_locations)} face(s): {known_count} known, {unknown_count} unknown"
    }


# ─── API Routes ───

@app.route("/detect", methods=["POST"])
def detect():
    """
    POST /detect
    Receives JPEG frame bytes in request body.
    Returns face recognition results as JSON.
    """
    try:
        frame_bytes = request.get_data()

        if not frame_bytes:
            return jsonify({
                "error": "No frame data received",
                "faces_detected": 0,
                "threat": False
            }), 400

        # Run face recognition
        result = recognize_faces(frame_bytes)

        # Add metadata
        result["server"] = "face_recognition"
        result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result["known_faces_loaded"] = len(known_face_names)

        return jsonify(result), 200

    except Exception as e:
        return jsonify({
            "error": str(e),
            "faces_detected": 0,
            "threat": False,
            "server": "face_recognition"
        }), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "server": "face_recognition",
        "port": 5002,
        "known_faces_loaded": len(known_face_names),
        "known_people": known_face_names,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }), 200


@app.route("/reload", methods=["POST"])
def reload_faces():
    """Reload known faces from the known_faces/ directory."""
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []

    print("\n🔄 Reloading known faces...")
    load_known_faces()

    return jsonify({
        "message": f"Reloaded. {len(known_face_names)} known face(s) loaded.",
        "known_people": known_face_names,
        "server": "face_recognition"
    }), 200


if __name__ == "__main__":
    print("=" * 50)
    print("🟣 EC2-2 — Face Recognition Server")
    print("=" * 50)

    # Load known faces at startup
    print(f"\n📂 Loading known faces from: {KNOWN_FACES_DIR}")
    load_known_faces()

    print(f"\n📡 Running on port 5002")
    print(f"🔗 POST /detect  → Send JPEG frame for face recognition")
    print(f"🔗 GET  /health  → Health check + loaded faces info")
    print(f"🔗 POST /reload  → Reload known faces from directory")
    print("=" * 50)

    app.run(host="0.0.0.0", port=5002, debug=False)
