FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for dlib + OpenCV + face_recognition
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create known_faces directory
RUN mkdir -p known_faces

# Copy known faces if they exist
COPY known_faces/ known_faces/

# Expose port
EXPOSE 5002

# Run the Flask server
CMD ["python", "app.py"]
