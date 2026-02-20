"""
Animal pose estimation using MMPose + AP-10K (HRNet-w32).
Two-stage pipeline:
1. YOLO detects animal bounding boxes
2. HRNet-w32 (AP-10K) estimates 17 keypoints per animal

AP-10K keypoints:
0=L_Eye, 1=R_Eye, 2=Nose, 3=Neck, 4=Root_of_tail
5=L_Shoulder, 6=L_Elbow, 7=L_F_Paw, 8=R_Shoulder, 9=R_Elbow
10=R_F_Paw, 11=L_Hip, 12=L_Knee, 13=L_B_Paw, 14=R_Hip, 15=R_Knee, 16=R_B_Paw

Drinking detection: keypoint 2 (Nose) Y-coordinate near water zone.
"""
import subprocess
import threading
import time
import os
import json
import cv2
import numpy as np

COOKIES_PATH = os.path.join(os.path.dirname(__file__), "cookies.txt")

KEYPOINT_NAMES = [
    "L_Eye", "R_Eye", "Nose", "Neck", "Root_of_tail",
    "L_Shoulder", "L_Elbow", "L_F_Paw", "R_Shoulder", "R_Elbow",
    "R_F_Paw", "L_Hip", "L_Knee", "L_B_Paw", "R_Hip", "R_Knee", "R_B_Paw",
]

SKELETON = [
    (0, 1), (0, 2), (1, 2), (2, 3), (3, 4),  # head to tail
    (3, 5), (5, 6), (6, 7),    # left front leg
    (3, 8), (8, 9), (9, 10),   # right front leg
    (4, 11), (11, 12), (12, 13),  # left back leg
    (4, 14), (14, 15), (15, 16),  # right back leg
]

COLORS = {
    "head": (0, 0, 255),     # Red for head keypoints
    "body": (0, 255, 255),   # Yellow for body
    "leg": (0, 255, 0),      # Green for legs
    "drinking": (255, 0, 255),  # Magenta for drinking indicator
}


def ensure_cookies():
    if not os.path.exists(COOKIES_PATH):
        cookies_env = os.environ.get("YT_COOKIES", "")
        if cookies_env:
            with open(COOKIES_PATH, "w") as f:
                f.write(cookies_env)


def get_stream_url(youtube_url: str) -> str:
    ensure_cookies()
    cmd = [
        "yt-dlp", "--cookies", COOKIES_PATH,
        "--js-runtimes", "node", "--remote-components", "ejs:github",
        "-f", "best[height<=720]", "-g", youtube_url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr.strip()[-200:]}")
    return result.stdout.strip().split("\n")[0]


class WateringHoleDetector:
    def __init__(self, youtube_url: str, water_zone=None):
        self.youtube_url = youtube_url
        # Water zone: polygon defining the water area (normalized 0-1 coords)
        # Default: center-bottom area typical for watering hole cams
        self.water_zone = water_zone or [(0.25, 0.5), (0.75, 0.5), (0.75, 0.85), (0.25, 0.85)]

        self.latest_annotated = None
        self.latest_detections = []
        self.latest_drinking = []
        self.fps = 0
        self.running = False
        self.error = None
        self._lock = threading.Lock()
        self._thread = None
        self._stream_url = None
        self._stream_url_time = 0

        # Lazy-load models
        self._det_model = None
        self._pose_inferencer = None

    def _load_models(self):
        from ultralytics import YOLO
        print("[detector] Loading YOLO detector...")
        self._det_model = YOLO('yolov8n.pt')

        print("[detector] Loading MMPose AP-10K model...")
        from mmpose.apis import MMPoseInferencer
        self._pose_inferencer = MMPoseInferencer(
            pose2d='td-hm_hrnet-w32_8xb64-210e_ap10k-256x256',
        )
        print("[detector] Models loaded")

    def _resolve_stream(self):
        now = time.time()
        if self._stream_url and (now - self._stream_url_time) < 1800:
            return self._stream_url
        print("[detector] Resolving stream URL...")
        self._stream_url = get_stream_url(self.youtube_url)
        self._stream_url_time = now
        return self._stream_url

    def _is_nose_in_water(self, nose_x, nose_y, img_w, img_h):
        """Check if nose keypoint is within the water zone polygon."""
        nx, ny = nose_x / img_w, nose_y / img_h
        # Simple point-in-polygon (ray casting)
        poly = self.water_zone
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > ny) != (yj > ny)) and (nx < (xj - xi) * (ny - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def _process_frame(self, frame):
        h, w = frame.shape[:2]
        ANIMAL_IDS = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23}

        # Stage 1: Detect animals with YOLO
        det_results = self._det_model(frame, conf=0.2, verbose=False)
        animal_bboxes = []
        for r in det_results:
            for box in r.boxes:
                if int(box.cls[0]) in ANIMAL_IDS:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    name = self._det_model.names[int(box.cls[0])]
                    animal_bboxes.append({"bbox": [x1, y1, x2, y2], "conf": conf, "class": name})

        # Stage 2: Run pose estimation on detected animals
        annotated = frame.copy()
        detections = []
        drinking = []

        if animal_bboxes:
            # Run MMPose with pre-detected bboxes
            bboxes_for_pose = [[b["bbox"][0], b["bbox"][1], b["bbox"][2], b["bbox"][3]] for b in animal_bboxes]
            result_gen = self._pose_inferencer(
                frame, bboxes=bboxes_for_pose, show=False, return_vis=False
            )

            for result in result_gen:
                preds = result['predictions'][0]
                for i, pred in enumerate(preds):
                    kps = pred['keypoints']
                    scores = pred['keypoint_scores']
                    bbox = animal_bboxes[i] if i < len(animal_bboxes) else {"bbox": [0,0,0,0], "conf": 0, "class": "unknown"}

                    nose_x, nose_y = kps[2][0], kps[2][1]
                    nose_conf = scores[2]
                    is_drinking = False

                    if nose_conf > 0.3:
                        is_drinking = self._is_nose_in_water(nose_x, nose_y, w, h)

                    det = {
                        "class": bbox["class"],
                        "confidence": bbox["conf"],
                        "bbox": bbox["bbox"],
                        "nose": [round(nose_x, 1), round(nose_y, 1)],
                        "nose_conf": round(nose_conf, 3),
                        "drinking": is_drinking,
                        "keypoints": [[round(kps[j][0],1), round(kps[j][1],1), round(scores[j],3)] for j in range(17)],
                    }
                    detections.append(det)
                    if is_drinking:
                        drinking.append(det)

                    # Draw skeleton
                    for j, (x, y) in enumerate(kps):
                        if scores[j] < 0.3:
                            continue
                        color = COLORS["head"] if j <= 2 else COLORS["body"] if j <= 4 else COLORS["leg"]
                        cv2.circle(annotated, (int(x), int(y)), 4, color, -1)

                    # Draw skeleton lines
                    for (a, b) in SKELETON:
                        if scores[a] > 0.3 and scores[b] > 0.3:
                            pt1 = (int(kps[a][0]), int(kps[a][1]))
                            pt2 = (int(kps[b][0]), int(kps[b][1]))
                            cv2.line(annotated, pt1, pt2, (255, 255, 0), 2)

                    # Draw nose marker (big circle if drinking)
                    if nose_conf > 0.3:
                        color = COLORS["drinking"] if is_drinking else COLORS["head"]
                        radius = 10 if is_drinking else 6
                        cv2.circle(annotated, (int(nose_x), int(nose_y)), radius, color, -1)
                        if is_drinking:
                            cv2.putText(annotated, "DRINKING", (int(nose_x) + 15, int(nose_y)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["drinking"], 2)

                    # Draw bbox
                    bx1, by1, bx2, by2 = bbox["bbox"]
                    cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

        # Draw water zone
        wz_pts = np.array([(int(x * w), int(y * h)) for x, y in self.water_zone], np.int32)
        cv2.polylines(annotated, [wz_pts], True, (255, 128, 0), 2)
        cv2.putText(annotated, "WATER ZONE", (wz_pts[0][0], wz_pts[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)

        return annotated, detections, drinking

    def _capture_loop(self):
        self._load_models()

        while self.running:
            try:
                url = self._resolve_stream()
                self.error = None
                cap = cv2.VideoCapture(url)
                if not cap.isOpened():
                    self.error = "Failed to open stream"
                    time.sleep(5)
                    self._stream_url = None
                    continue

                print("[detector] Stream opened")
                last_detect = 0
                frame_interval = 2.0  # seconds

                while self.running and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    now = time.time()
                    if now - last_detect < frame_interval:
                        continue
                    last_detect = now

                    t0 = time.time()
                    annotated, detections, drinking = self._process_frame(frame)
                    dt = time.time() - t0

                    with self._lock:
                        self.latest_annotated = annotated
                        self.latest_detections = detections
                        self.latest_drinking = drinking
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
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=10)

    def get_annotated_jpeg(self, quality=80):
        with self._lock:
            if self.latest_annotated is None:
                return None
            _, buf = cv2.imencode(".jpg", self.latest_annotated, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return buf.tobytes()

    def get_state(self):
        with self._lock:
            return {
                "detections": self.latest_detections,
                "count": len(self.latest_detections),
                "drinking_count": len(self.latest_drinking),
                "drinking": self.latest_drinking,
                "fps": self.fps,
                "timestamp": time.time(),
                "error": self.error,
            }
