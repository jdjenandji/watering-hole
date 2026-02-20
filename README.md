# 🦒 Watering Hole Tracker

Real-time animal detection at the [Africam Naledi Cat-EYE watering hole](https://www.youtube.com/live/ydYDqZQpim8) using YOLOv8.

![Status](https://img.shields.io/badge/status-phase%201-blue)

## What It Does

Captures frames from the YouTube livestream, runs YOLOv8 object detection to identify animals (elephants, zebras, giraffes, birds, etc.), and displays annotated frames with bounding boxes in a web dashboard.

## Setup

### Prerequisites
- Python 3.10+
- `ffmpeg` installed
- `yt-dlp` installed

### Install

```bash
cd backend
pip install -r requirements.txt
```

### Run

```bash
cd backend
python server.py
```

Open `http://localhost:8000` in your browser.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `YOUTUBE_URL` | Naledi cam | YouTube livestream URL |
| `MODEL_SIZE` | `yolov8n.pt` | YOLOv8 model (n/s/m/l/x) |
| `CONFIDENCE` | `0.3` | Min detection confidence |

## Architecture

```
YouTube Livestream
       │
       ▼ (yt-dlp resolves stream URL)
   OpenCV VideoCapture
       │
       ▼ (~1 frame/sec)
   YOLOv8 Detection
       │
       ▼ (filter animal classes only)
  FastAPI WebSocket
       │
       ▼ (base64 JPEG + JSON)
   Browser Dashboard
```

## Animal Classes Detected

🐘 Elephant · 🦓 Zebra · 🦒 Giraffe · 🐄 Cow · 🐴 Horse · 🐑 Sheep · 🐻 Bear · 🐦 Bird · 🐱 Cat · 🐕 Dog

Uses COCO pretrained weights — works out of the box for common African wildlife. Custom training for specific species (impala, kudu, warthog, etc.) is planned for Phase 2.

## Roadmap

- [x] **Phase 1**: Real-time bounding boxes around animals
- [ ] **Phase 2**: Species tracking + drink counting (when an animal's head dips to water level)
- [ ] **Phase 3**: Historical dashboard — daily/weekly animal visit stats
- [ ] **Phase 4**: Custom YOLOv8 model trained on African wildlife

## License

MIT
