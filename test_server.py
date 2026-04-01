"""
Test script for Face Recognition Server.
Sends sample frames to the server and verifies responses.
Run the server first: python app.py
"""

import requests
import cv2
import numpy as np

SERVER_URL = "http://localhost:5002"


def create_blank_frame(width=640, height=480):
    """Create a blank frame with no faces (just solid color)."""
    frame = np.full((height, width, 3), (80, 80, 80), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()


def capture_webcam_frame():
    """Capture a single frame from the webcam (for real face testing)."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    _, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()


def test_health():
    """Test health check endpoint."""
    print("=" * 50)
    print("🔍 Testing /health endpoint...")
    resp = requests.get(f"{SERVER_URL}/health")
    data = resp.json()
    print(f"   Status: {resp.status_code}")
    print(f"   Server: {data['server']}")
    print(f"   Known faces loaded: {data['known_faces_loaded']}")
    print(f"   Known people: {data['known_people']}")
    assert resp.status_code == 200
    print("   ✅ Health check passed!\n")


def test_no_face():
    """Send a blank frame — should detect no faces."""
    print("=" * 50)
    print("🔍 Testing NO FACE (blank frame)...")

    frame = create_blank_frame()
    resp = requests.post(f"{SERVER_URL}/detect", data=frame,
                         headers={"Content-Type": "image/jpeg"})
    result = resp.json()

    print(f"   Faces detected: {result['faces_detected']}")
    print(f"   Message: {result['message']}")
    print(f"   Threat: {result['threat']}")
    assert result["faces_detected"] == 0
    assert result["threat"] == False
    print("   ✅ No face test passed!\n")


def test_webcam_face():
    """Capture a frame from the webcam and test face detection."""
    print("=" * 50)
    print("🔍 Testing WEBCAM FACE (live capture)...")

    frame = capture_webcam_frame()
    if frame is None:
        print("   ⚠️  Could not open webcam — skipping test")
        return

    resp = requests.post(f"{SERVER_URL}/detect", data=frame,
                         headers={"Content-Type": "image/jpeg"})
    result = resp.json()

    print(f"   Faces detected: {result['faces_detected']}")
    print(f"   Message: {result['message']}")
    print(f"   Threat: {result['threat']}")

    if result["faces_detected"] > 0:
        for face in result["faces"]:
            icon = "🟢" if face["name"] != "Unknown" else "🔴"
            print(f"   {icon} {face['name']} (confidence: {face['confidence']})")

    print("   ✅ Webcam face test completed!\n")


def test_reload():
    """Test reload known faces endpoint."""
    print("=" * 50)
    print("🔍 Testing /reload endpoint...")
    resp = requests.post(f"{SERVER_URL}/reload")
    data = resp.json()
    print(f"   Status: {resp.status_code}")
    print(f"   Message: {data['message']}")
    print(f"   Known people: {data['known_people']}")
    assert resp.status_code == 200
    print("   ✅ Reload test passed!\n")


def test_empty_body():
    """Send empty request body — should return 400."""
    print("=" * 50)
    print("🔍 Testing EMPTY BODY (error handling)...")

    resp = requests.post(f"{SERVER_URL}/detect", data=b"",
                         headers={"Content-Type": "image/jpeg"})
    print(f"   Status: {resp.status_code}")
    assert resp.status_code == 400
    print("   ✅ Empty body error handling passed!\n")


if __name__ == "__main__":
    print("\n🚀 Face Recognition Server — Test Suite")
    print("=" * 50)
    print(f"   Target: {SERVER_URL}\n")

    try:
        test_health()
        test_empty_body()
        test_no_face()
        test_webcam_face()
        test_reload()

        print("=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 50)

    except requests.ConnectionError:
        print("❌ Could not connect to server. Make sure it's running:")
        print("   python app.py")
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
