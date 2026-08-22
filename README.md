# 🚦 Traffic Surveillance & Intelligence System

> **Don't just watch traffic. Turn traffic into structured, searchable traffic data.**

![Live Traffic Monitoring](images/main-dashboard-1.png)

![Traffic Operations Analytics](images/analytics-dashboard.png)

![Live Detections & Evidence Feed](images/main-dashboard-2.png)

---

## 💡 The Core Philosophy

A conventional traffic camera simply records pixels:

> *"There are vehicles on the road."*

This system transforms a fixed traffic camera into an automated, continuous source of **structured traffic intelligence**.

Instead of treating video as ephemeral footage, the pipeline extracts and structures every observation in real time:
- **Vehicle Classification & Tracking**: Unique track ID, vehicle class (car, motorcycle, bus, truck, auto-rickshaw).
- **Spatial Lane Assignment**: Exact perspective polygon localization using ground-contact coordinates.
- **Directional Classification**: Top-to-Bottom (`T2B`) vs Bottom-to-Top (`B2T`) flow analysis.
- **Traffic Rule Violations**: Direction enforcement (wrong-way driving) and unauthorized cycle/pedestrian zone intrusion.
- **Continuous Flow Metrics**: Vehicle velocity vectors, duration in scene, and volume over time.

### Why Structured Data Matters
Video files are large, unindexed, and temporary. Structured event records in a database are lightweight, permanent, and instantly queryable.

With this architecture, the system can instantly answer critical operational questions:
- *How is traffic volume distributed between T2B and B2T directions?*
- *Which specific lane experiences the highest congestion during peak hours?*
- *What percentage of the fleet consists of two-wheelers vs heavy commercial vehicles?*
- *When and where did specific traffic rule violations occur, and what is the visual evidence?*

---

## 🏛️ System Architecture

```text
                             ┌───────────────────────────────┐
                             │      Fixed Camera Stream      │
                             │     (traffic_2.mp4, RTSP)     │
                             └───────────────┬───────────────┘
                                             │
                                             ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                              COMPUTER VISION PIPELINE                                  │
 ├────────────────────────────────────────────────────────────────────────────────────────┤
 │  1. Video Decode         │ cv2.VideoCapture (Frame Buffering)                          │
 │  2. Vehicle Detection    │ YOLOv11 (FP16 Tensor Core Accelerated on NVIDIA CUDA)       │
 │  3. Multi-Object Tracker │ ByteTrack / SORT Matching (min_hits=1, max_age=30)          │
 │  4. Perspective Polygons │ 4-Point ROI pointPolygonTest evaluated at Ground Contact    │
 │  5. Rule Enforcement     │ Multi-Frame Streak Verification (Wrong-Way & Zone Breach)   │
 │  6. Visual HUD Renderer  │ Real-Time Overlays, Alert Banners & Live Measured FPS Badge │
 └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │  (Asynchronous Thread Queue)
                                             ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                           PERSISTENCE LAYER (SQLite WAL)                               │
 ├────────────────────────────────────────────────────────────────────────────────────────┤
 │  • WAL Mode (Write-Ahead Logging) + Transaction Batching for Non-Blocking 0.2ms I/O    │
 │  • Tables: vehicles, events (violations), counts, lane_stats                           │
 └───────────────────────────────┬───────────────────────┬────────────────────────────────┘
                                 │                       │
                                 ▼                       ▼
 ┌───────────────────────────────────────────┐ ┌──────────────────────────────────────────┐
 │           LIVE STREAM VIEW (/)            │ │     OPERATIONS ANALYTICS (/analytics)    │
 ├───────────────────────────────────────────┤ ├──────────────────────────────────────────┤
 │ • First-Frame Polygon Region Configurator │ │ • High-Frequency Polling (1.5s interval) │
 │ • Low-Latency MJPEG Live Stream           │ │ • Zero-Flicker In-Place Chart.js Updates │
 │ • Real-Time Measured FPS & Violation HUD  │ │ • Traffic Flow Timeline & Direction Gauge│
 │ • Interactive Pause / Stop Controls       │ │ • Lane Analytics & Vehicle Mix Breakdown │
 │                                           │ │ • Live Searchable Table & CSV Export     │
 └───────────────────────────────────────────┘ └──────────────────────────────────────────┘
```

---

## ✨ Key Features

### 1. 🎯 Interactive First-Frame Region Configuration
- Visual canvas on page load allowing operators to draw arbitrary 4-point perspective polygons directly on the road:
  - **Lane 1** (e.g., Direction `B2T`)
  - **Lane 2** (e.g., Direction `T2B`)
  - **Cycle / Pedestrian Zone**
  - **Custom Additional Lanes & Monitoring Zones**
- Configurations are saved to `output/lanes_config.json` and persist across sessions.

### 2. ⚡ Accelerated NVIDIA GPU (FP16) Inference
- **Tensor Core FP16 Optimization**: YOLOv11 model weights are initialized directly in half-precision on CUDA, reducing detection latency to **~35 ms**.
- **Asynchronous Decoupled Database Queue**: Database writes are batched in a background daemon thread with SQLite WAL mode. Main CV loop disk I/O latency dropped from **615 ms to 0.29 ms** (~25x overall throughput improvement).
- **True Measured FPS**: Real rolling-window FPS computed and displayed live on the video HUD and dashboard.

### 3. 🚨 Ground-Contact Traffic Rule Violation Engine
- **Ground-Plane Contact Geometry**: All polygon tests evaluate the center of the bottom bounding box edge `(cx, y2)` (tire/road contact point) rather than the centroid, preventing perspective false-positives.
- **Wrong-Way Driving Enforcement**: Evaluates vehicle motion vectors inside directional lanes. Vehicles moving Top-to-Bottom in a Bottom-to-Top lane (or vice versa) are flagged.
- **Cycle & Pedestrian Zone Intrusion**: Multi-wheel vehicles (`car`, `truck`, `bus`, `van`) entering designated cycle/pedestrian zones trigger instant intrusion alerts.
- **Multi-Frame Noise Filter**: Requires persistent displacement (`|dy| >= 18px` over 8+ frames) verified across 4 consecutive confirmation frames to eliminate detector jitter false alarms.

### 4. 📊 Operations Analytics Center (`/analytics`)
- **Zero Page-Reload Architecture**: Live data dashboard polling SQLite every 1.5 seconds with in-place Chart.js updates without canvas flickering.
- **Executive KPIs**: Total volume, session increment, directional breakdown (`T2B` vs `B2T`), and violation rates.
- **Directional Traffic Balance**: High-contrast visual split gauge indicating flow dominance.
- **Traffic Volume Timeline**: Area chart visualizing vehicle count over time with distinct series for `T2B` (Sky Blue) and `B2T` (Emerald Green).
- **Lane Analytics**: Stacked horizontal bar chart detailing traffic volume by lane and direction.
- **Vehicle Mix**: Donut chart and dense tabular count for Cars, Motorcycles, Buses, and Trucks.
- **Searchable Vehicle Activity & CSV Export**: Real-time filterable table of all tracked vehicles with instantaneous CSV report export.

### 5. 🔍 Number Plate (ANPR) Pipeline Status
- The system includes a modular ANPR subsystem (`PlateDetector` + `ANPREngine`).
- **Honest Empirical Evaluation**: Evaluated across 300 frames of the elevated surveillance camera footage (`traffic_2.mp4`). Because the camera is a fixed wide-angle elevated overview, license plates average only **52.9 × 23.4 pixels** with perspective tilt and motion blur, which is insufficient for reliable character-level OCR without telephoto optical zoom.
- **Design Decision**: ANPR is decoupled from the main inference path to avoid generating guessed plate strings and preserve high FPS. The modular interface is preserved for integration with dedicated close-up ANPR camera feeds.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA support (CPU fallback is supported automatically)

### 1. Clone the Repository
```bash
git clone https://github.com/priyanshu596/Traffic_survelliance.git
cd Traffic_survelliance
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python3 -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download Model Weights & Video
Place your YOLO weights in the `models/` directory:
- `models/UVH-26-MV-YOLOv11-S.pt`
- `models/license-plate-finetune-v1l.pt` (optional)

Place your video file in the project root (e.g. `traffic_2.mp4`).

### 4. Launch the Interactive Dashboard
```bash
python dashboard.py -v traffic_2.mp4
```

### 5. Open in Your Browser
- **Live Stream & Region Drawer**: [http://localhost:5000](http://localhost:5000)
  1. Click 4 corner points around each lane on the first frame.
  2. Click **"🚀 START INFERENCE"**.
- **Operations Analytics Dashboard**: [http://localhost:5000/analytics](http://localhost:5000/analytics)
  - Monitor real-time traffic flow curves, lane distributions, violation feeds, and vehicle history.

---

## 🛠️ CLI Options

```bash
python dashboard.py --help
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `-v`, `--video` | string | `traffic_2.mp4` | Path to input traffic video file |
| `--vehicle-model` | string | `models/UVH-26-MV-YOLOv11-S.pt` | Path to YOLOv11 vehicle detector weights |
| `--device` | string | `cuda` (or `cpu`) | PyTorch device target (`cuda`, `0`, `cpu`) |
| `--conf` | float | `0.25` | Confidence detection threshold |
| `--iou` | float | `0.45` | IoU NMS threshold |
| `--port` | int | `5000` | Web dashboard port |
| `--no-annotated` | flag | `False` | Skip saving output annotated video to disk |

---

## 📂 Project Structure

```text
Traffic_survelliance/
├── config.py                     # Central configuration & lane data structures
├── dashboard.py                  # Main application entry point
├── requirements.txt              # Production package requirements
├── .gitignore                    # Git exclusion rules
├── README.md                     # Comprehensive documentation
├── images/                       # Dashboard screenshots for documentation
│   ├── main-dashboard-1.png
│   ├── analytics-dashboard.png
│   └── main-dashboard-2.png
├── models/                       # YOLO model weights directory
│   ├── UVH-26-MV-YOLOv11-S.pt
│   └── license-plate-finetune-v1l.pt
├── src/
│   ├── classification/           # Vehicle category classification
│   │   └── vehicle_classifier.py
│   ├── counting/                 # Polygon lane & direction accumulators
│   │   ├── direction_counter.py
│   │   └── lane_counter.py
│   ├── dashboard/                # Flask web server & live operations UI
│   │   └── app.py
│   ├── detection/                # YOLO GPU vehicle & plate detectors
│   │   ├── plate_detector.py
│   │   └── vehicle_detector.py
│   ├── monitors/                 # Traffic rules & violation monitor
│   │   ├── pedestrian_cyclist.py
│   │   └── traffic_rules.py
│   ├── ocr/                      # Modular ANPR engine
│   │   └── anpr_engine.py
│   ├── segmentation/             # Instance segmentation module
│   ├── storage/                  # Asynchronous SQLite WAL event storage
│   │   └── event_storage.py
│   ├── tracking/                 # Multi-object tracker (SORT/ByteTrack)
│   │   └── multi_object_tracker.py
│   ├── visualizer/               # Visual evidence generator & HUD
│   │   └── evidence.py
│   └── pipeline.py               # End-to-end CV orchestration pipeline
└── output/                       # Generated outputs, configs & databases
    ├── lanes_config.json         # User-defined lane & zone polygons
    └── traffic_events.db         # Structured SQLite analytics database
```

---

## 🔮 Future Roadmap: AI / VLM Natural Language Query Layer

Because every vehicle detection and traffic event is converted into indexed, structured tabular data in SQLite, the next evolution of this system is integrating a **Vision-Language Model (VLM) / LLM Agent layer**.

Operators will be able to query the camera using plain English:
- *"How many heavy trucks entered Lane 1 between 2:00 PM and 3:00 PM?"*
- *"Show me visual evidence of all wrong-way driving events from yesterday."*
- *"What was the peak traffic volume hour in the Top-to-Bottom direction?"*

```text
                     ┌────────────────────────────────────────┐
                     │    Natural Language Query Interface    │
                     │  "Show me wrong-way vehicles at 2 PM"  │
                     └───────────────────┬────────────────────┘
                                         │
                                         ▼
                     ┌────────────────────────────────────────┐
                     │         LLM / Text-to-SQL Agent        │
                     └───────────────────┬────────────────────┘
                                         │
                                         ▼
                     ┌────────────────────────────────────────┐
                     │   Structured SQLite Traffic Database   │
                     └────────────────────────────────────────┘
```

---

## 🏁 Summary

> **"Don't just watch traffic. Turn traffic into data."**

This project bridges the gap between raw computer-vision bounding boxes and actionable traffic analytics, delivering a production-ready intelligence platform for smart cities and modern traffic operations centers.
