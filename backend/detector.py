"""
Frame capture from YouTube livestream + YOLOv8 animal detection.
"""
import subprocess
import threading
import time
import json
import cv2
import numpy as np
from ultralytics import YOLO

# COCO classes that are animals
ANIMAL_CLASSES = {
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep",
    19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe",
}
ANIMAL_IDS = set(ANIMAL_CLASSES.keys())

# Colors per species (BGR)
COLORS = {
    "bird": (0, 255, 255),
    "cat": (255, 0, 255),
    "dog": (255, 165, 0),
    "horse": (0, 165, 255),
    "sheep": (200, 200, 200),
    "cow": (100, 100, 255),
    "elephant": (128, 128, 128),
    "bear": (50, 50, 200),
    "zebra": (255, 255, 255),
    "giraffe": (0, 200, 200),
}


def get_stream_url(youtube_url: str) -> str:
    """Use yt-dlp to resolve the actual stream URL."""
    result = subprocess.run(
        ["yt-dlp", "-f", "best[height<=720]", "-g", youtube_url],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr.strip()}")
    return result.stdout.strip().split("\n")[0]


class AnimalDetector:
    def __init__(self, youtube_url: str, model_size: str = "yolov8n.pt", conf_threshold: float = 0.3):
        self.youtube_url = youtube_url
        self.conf_threshold = conf_threshold
        self.model = YOLO(model_size)
        self.latest_frame = None
        self.latest_annotated = None
        self.latest_detections = []
        self.fps = 0
        self.running = False
        self._lock = threading.Lock()
        self._capture_thread = None
        self._stream_url = None
        self._stream_url_time = 0

    def _resolve_stream(self):
        """Resolve stream URL, cache for 30 min."""
        now = time.time()
        if self._stream_url and (now - self._stream_url_time) < 1800:
            return self._stream_url
        print("Resolving YouTube stream URL...")
        self._stream_url = get_stream_url(self.youtube_url)
        self._stream_url_time = now
        print(f"Stream URL resolved: {self._stream_url[:80]}...")
        return self._stream_url

    def _capture_loop(self):
        """Main capture + detection loop."""
        while self.running:
            try:
                url = self._resolve_stream()
                cap = cv2.VideoCapture(url)
                if not cap.isOpened():
                    print("Failed to open stream, retrying in 5s...")
                    time.sleep(5)
                    self._stream_url = None  # Force re-resolve
                    continue

                print("Stream opened, starting detection...")
                frame_interval = 1.0  # seconds between detections
                last_detect = 0

                while self.running and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        print("Frame read failed, reconnecting...")
                        break

                    now = time.time()
                    if now - last_detect < frame_interval:
                        continue
                    last_detect = now

                    # Run detection
                    t0 = time.time()
                    results = self.model(frame, verbose=False, conf=self.conf_threshold)
                    dt = time.time() - t0

                    # Filter to animals only
                    detections = []
                    annotated = frame.copy()

                    for r in results:
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            if cls_id not in ANIMAL_IDS:
                                continue
                            conf = float(box.conf[0])
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            label = ANIMAL_CLASSES[cls_id]
                            color = COLORS.get(label, (0, 255, 0))

                            detections.append({
                                "class": label,
                                "confidence": round(conf, 3),
                                "bbox": [x1, y1, x2, y2],
                            })

                            # Draw on frame
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                            text = f"{label} {conf:.0%}"
                            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                            cv2.putText(annotated, text, (x1 + 2, y1 - 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

                    with self._lock:
                        self.latest_frame = frame
                        self.latest_annotated = annotated
                        self.latest_detections = detections
                        self.fps = round(1 / dt, 1) if dt > 0 else 0

                cap.release()

            except Exception as e:
                print(f"Error in capture loop: {e}")
                time.sleep(5)

    def start(self):
        """Start the capture + detection thread."""
        if self.running:
            return
        self.running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        print("Detector started")

    def stop(self):
        """Stop the capture thread."""
        self.running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=10)
        print("Detector stopped")

    def get_annotated_jpeg(self, quality: int = 80) -> bytes | None:
        """Get the latest annotated frame as JPEG bytes."""
        with self._lock:
            if self.latest_annotated is None:
                return None
            _, buf = cv2.imencode(".jpg", self.latest_annotated, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return buf.tobytes()

    def get_state(self) -> dict:
        """Get current detection state as JSON-serializable dict."""
        with self._lock:
            return {
                "detections": self.latest_detections,
                "count": len(self.latest_detections),
                "fps": self.fps,
                "timestamp": time.time(),
            }
