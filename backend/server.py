"""
FastAPI server: serves annotated frames via WebSocket + static frontend.
"""
import asyncio
import base64
import json
import os
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from detector import AnimalDetector

YOUTUBE_URL = os.environ.get("YOUTUBE_URL", "https://www.youtube.com/live/ydYDqZQpim8")
MODEL_SIZE = os.environ.get("MODEL_SIZE", "yolov8n.pt")
CONFIDENCE = float(os.environ.get("CONFIDENCE", "0.3"))

app = FastAPI(title="Watering Hole Tracker")
detector = AnimalDetector(YOUTUBE_URL, MODEL_SIZE, CONFIDENCE)

# Serve frontend
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")


@app.on_event("startup")
async def startup():
    detector.start()


@app.on_event("shutdown")
async def shutdown():
    detector.stop()


@app.get("/")
async def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/api/status")
async def status():
    state = detector.get_state()
    return {
        "running": detector.running,
        "stream_url": YOUTUBE_URL,
        **state,
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("WebSocket client connected")
    try:
        while True:
            # Get annotated frame
            jpeg = detector.get_annotated_jpeg()
            state = detector.get_state()

            if jpeg:
                # Send frame as base64 + detections as JSON
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
                await ws.send_text(json.dumps({"type": "waiting", "message": "Waiting for first frame..."}))

            await asyncio.sleep(1)  # ~1 fps updates

    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
