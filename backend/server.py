"""
FastAPI server: serves annotated frames via WebSocket + static frontend.
"""
import asyncio
import base64
import json
import os
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

try:
    from pose_detector import WateringHoleDetector
    USE_POSE = True
except ImportError:
    from detector import AnimalDetector
    USE_POSE = False

YOUTUBE_URL = os.environ.get("YOUTUBE_URL", "https://www.youtube.com/live/ydYDqZQpim8")

app = FastAPI(title="Watering Hole Tracker")
if USE_POSE:
    detector = WateringHoleDetector(YOUTUBE_URL)
else:
    detector = AnimalDetector(YOUTUBE_URL)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")


@app.on_event("startup")
async def startup():
    detector.start()


@app.on_event("shutdown")
async def shutdown():
    detector.stop()


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/")
async def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/api/status")
async def status():
    state = detector.get_state()
    return {"running": detector.running, "stream_url": YOUTUBE_URL, **state}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("[ws] Client connected")
    try:
        while True:
            jpeg = detector.get_annotated_jpeg()
            state = detector.get_state()

            if jpeg:
                payload = {
                    "type": "frame",
                    "image": base64.b64encode(jpeg).decode("ascii"),
                    "detections": state["detections"],
                    "count": state["count"],
                    "fps": state["fps"],
                    "timestamp": state["timestamp"],
                }
                await ws.send_text(json.dumps(payload))
            else:
                msg = state.get("error") or "Connecting to stream..."
                await ws.send_text(json.dumps({"type": "waiting", "message": msg}))

            await asyncio.sleep(1.5)

    except WebSocketDisconnect:
        print("[ws] Client disconnected")
    except Exception as e:
        print(f"[ws] Error: {e}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
