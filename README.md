# EC2-2 — Face Recognition Server

Flask server that detects and recognizes faces in surveillance frames using the `face_recognition` library (powered by dlib).

## How It Works

1. At startup, loads all face photos from `known_faces/` directory and computes 128-d encodings
2. Receives JPEG frame bytes via `POST /detect`
3. Detects face locations using **HOG detector** (CPU-friendly)
4. Computes **128-dimensional face encodings** for each detected face
5. Compares each encoding against the known-faces DB with a tolerance threshold (0.6)
6. Labels each face as **Known (name)** or **Unknown** — unknown faces trigger a `threat: true`

## API Endpoints

| Method | Route     | Description                                      |
|--------|-----------|--------------------------------------------------|
| POST   | `/detect` | Send JPEG frame → Get face recognition results   |
| GET    | `/health` | Health check + list of loaded known faces         |
| POST   | `/reload` | Reload known faces without restarting server      |

## Response Format

```json
{
  "faces_detected": 2,
  "faces": [
    {"name": "Devashish", "confidence": 0.8234, "location": {"top": 80, "right": 300, "bottom": 240, "left": 160}},
    {"name": "Unknown", "confidence": 0.3102, "location": {"top": 90, "right": 520, "bottom": 250, "left": 400}}
  ],
  "unknown_count": 1,
  "known_count": 1,
  "threat": true,
  "message": "Detected 2 face(s): 1 known, 1 unknown",
  "server": "face_recognition",
  "timestamp": "2026-04-01 23:45:00",
  "known_faces_loaded": 3
}
```

## Adding Known Faces

Place photos in the `known_faces/` folder:

```
known_faces/
├── devashish.jpg        → labeled as "Devashish"
├── john_doe.jpg         → labeled as "John Doe"
└── alice_sharma.png     → labeled as "Alice Sharma"
```

- **One clear face** per photo
- Filename becomes the person's name (underscores → spaces, auto title-cased)
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`
- Use `POST /reload` to add new faces without restarting

## Local Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Add known face photos
cp your_photo.jpg known_faces/devashish.jpg

# Run the server
python app.py
```

## EC2 Deployment

```bash
# SSH into EC2 instance
ssh -i key.pem ubuntu@<ec2-ip>

# Install system deps (needed for dlib compilation)
sudo apt update
sudo apt install -y python3-pip cmake build-essential

# Install Python deps
pip install -r requirements.txt

# Upload known_faces photos via SCP
scp -i key.pem your_photo.jpg ubuntu@<ec2-ip>:~/face_recognition_server/known_faces/

# Run server (background)
nohup python3 app.py &

# Or use Docker
docker build -t face-recognition .
docker run -d -p 5002:5002 face-recognition
```

## EC2 Security Group

Open **port 5002** for inbound TCP traffic.

## Tech Stack

- Python 3.11
- Flask
- face_recognition (dlib)
- OpenCV (headless)
- NumPy

## Note

> dlib compilation requires `cmake` and `build-essential` on Ubuntu. On t2.micro, the build can take 5-10 minutes. Docker handles this automatically.
