FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 curl gcc g++ git \
    && rm -rf /var/lib/apt/lists/*

# Node.js for yt-dlp EJS
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch CPU first
RUN pip install --no-cache-dir torch==2.1.0+cpu torchvision==0.16.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install mmcv with CUDA-free build
RUN pip install --no-cache-dir openmim && \
    mim install mmcv==2.1.0

# Install mmpose, mmdet, and other deps
RUN pip install --no-cache-dir mmpose mmdet

COPY backend/requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Pre-download model weights
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
RUN python -c "from mmpose.apis import MMPoseInferencer; MMPoseInferencer(pose2d='td-hm_hrnet-w32_8xb64-210e_ap10k-256x256')" || true

COPY backend/ ./backend/
COPY frontend/ ./frontend/

WORKDIR /app/backend
EXPOSE 8000
CMD ["python", "server.py"]
