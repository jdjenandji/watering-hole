"""
Frame capture from YouTube livestream + YOLOv8 animal detection.
"""
import subprocess
import threading
import time
import os
import cv2
import numpy as np
from ultralytics import YOLO

# COCO classes that are animals
ANIMAL_CLASSES = {
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep",
    19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe",
}
ANIMAL_IDS = set(ANIMAL_CLASSES.keys())

COLORS = {
    "bird": (0, 255, 255), "cat": (255, 0, 255), "dog": (255, 165, 0),
    "horse": (0, 165, 255), "sheep": (200, 200, 200), "cow": (100, 100, 255),
    "elephant": (128, 128, 128), "bear": (50, 50, 200),
    "zebra": (255, 255, 255), "giraffe": (0, 200, 200),
}

COOKIES_PATH = os.path.join(os.path.dirname(__file__), "cookies.txt")


def ensure_cookies():
    """Write cookies from env var if file doesn't exist."""
    if not os.path.exists(COOKIES_PATH):
        cookies_env = os.environ.get("YT_COOKIES", "")
        if cookies_env:
            with open(COOKIES_PATH, "w") as f:
                f.write(cookies_env)
            print(f"[detector] Wrote cookies from env ({len(cookies_env)} chars)")
        else:
            print("[detector] WARNING: No cookies.txt and no YT_COOKIES env var!")


def get_stream_url(youtube_url: str) -> str:
    """Use yt-dlp with cookies + EJS to resolve the actual stream URL."""
    ensure_cookies()
    cmd = [
        "yt-dlp",
        "--cookies", COOKIES_PATH,
        "--js-runtimes", "node",
        "--remote-components", "ejs:github",
        "-f", "best[height<=720]",
        "-g", youtube_url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr.strip()[-200:]}")
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
        self.error = None
        self._lock = threading.Lock()
        self._capture_thread = None
        self._stream_url = None
        self._stream_url_time = 0

    def _resolve_stream(self):
        now = time.time()
        if self._stream_url and (now - self._stream_url_time) < 1800:
            return self._stream_url
        print("[detector] Resolving YouTube stream URL...")
        self._stream_url = get_stream_url(self.youtube_url)
        self._stream_url_time = now
        print(f"[detector] Got stream URL ({len(self._stream_url)} chars)")
        return self._stream_url

    def _capture_loop(self):
        while self.running:
            try:
                url = self._resolve_stream()
                self.error = None
                print("[detector] Opening stream with OpenCV...")
                cap = cv2.VideoCapture(url)
                if not cap.isOpened():
                    self.error = "Failed to open stream"
                    print(f"[detector] {self.error}, retrying in 5s...")
                    time.sleep(5)
                    self._stream_url = None
                    continue

                print("[detector] Stream opened, detecting...")
                frame_interval = 1.5
                last_detect = 0

                while self.running and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        print("[detector] Frame read failed, reconnecting...")
                        break

                    now = time.time()
                    if now - last_detect < frame_interval:
                        continue
                    last_detect = now

                    t0 = time.time()
                    results = self.model(frame, verbose=False, conf=self.conf_threshold)
                    dt = time.time() - t0

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
                self.error = str(e)
                print(f"[detector] Error: {e}")
                time.sleep(5)
                self._stream_url = None

    def start(self):
        if self.running:
            return
        self.running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

    def stop(self):
        self.running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=10)

    def get_annotated_jpeg(self, quality: int = 80) -> bytes | None:
        with self._lock:
            if self.latest_annotated is None:
                return None
            _, buf = cv2.imencode(".jpg", self.latest_annotated, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return buf.tobytes()

    def get_state(self) -> dict:
        with self._lock:
            return {
                "detections": self.latest_detections,
                "count": len(self.latest_detections),
                "fps": self.fps,
                "timestamp": time.time(),
                "error": self.error,
            }
