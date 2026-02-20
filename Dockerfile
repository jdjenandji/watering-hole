FROM python:3.11-slim

# System deps: ffmpeg for video, libgl for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLO model weights
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

COPY backend/ ./backend/
COPY frontend/ ./frontend/

WORKDIR /app/backend

EXPOSE 8000

CMD ["python", "server.py"]
