# CivicPulse AI: Autonomous Multimodal Civic Grievance & Infrastructure Triage Engine

[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![YOLOv8](https://img.shields.io/badge/Vision-Ultralytics%20YOLOv8-FF7F00.svg)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Whisper](https://img.shields.io/badge/Audio-OpenAI%20Whisper-00A67E.svg)](https://github.com/openai/whisper)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9+-5C3EE8.svg?logo=OpenCV&logoColor=white)](https://opencv.org/)
[![Leaflet](https://img.shields.io/badge/Geospatial-Leaflet.js-199900.svg)](https://leafletjs.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

CivicPulse AI is an enterprise-grade, autonomous civic grievance triage system. It continuously ingests multimodal citizen reports (**high-resolution photographs, motion videos, Web Audio API voice dispatches, and live HTML5 GPS coordinates**), dynamically analyzes defects using a **3-Layer Computer Vision Pipeline**, enriches incident context using live OpenStreetMap and Open-Meteo REST APIs, prevents false emergency escalations with **Visual Evidence Integrity Guards**, and dispatches actionable tickets to municipal departments with dynamic SLA countdowns.

---

## 🏛️ System Architecture

```
[ Citizen Smartphone / Web Dashboard ]
       │
       ├── 1. Live HTML5 Geolocation (navigator.geolocation) & Interactive Map Pin
       ├── 2. Defect Evidence: Photograph OR Motion Video (.mp4, .webm, .mov, .avi)
       └── 3. Citizen Voice Dispatch (Web Audio API Recorder)
                 │
                 ▼
     [ FastAPI Ingestion Gateway (/api/v1/reports) ]
                 │
  ┌──────────────┼──────────────┬──────────────┬───────────────────────────────┐
  ▼              ▼              ▼              ▼                               ▼
[ EXIF & dHash ] [ Live OSM ]  [ Open-Meteo ]  [ 3-Layer Vision Engine ]     [ Audio Dispatch ]
Perceptual Hash  Nominatim      Live Weather   1. YOLOv8 (Civic / RDD2022)   Whisper Multi-Lingual
Anti-Spoofing    Reverse-Geo    Precipitation  2. MobileNetV2 CNN Classifier Speech-to-Text &
Fingerprinting   Ward & Street  Risk Score     3. OpenCV Adaptive Contours   Urgency Extraction
  │              │              │              │                               │
  └──────────────┴──────────────┴──────────────┼───────────────────────────────┘
                                               ▼
                         [ Multimodal Severity Formulation Engine ]
              S = clip(0.30·C_conf + 0.35·R_area + 0.20·W_hazard + 0.15·E_weather, 0, 1)
               * Includes Visual Integrity Guard (prevents false emergency triage)
                                               │
                                               ▼
                         [ Spatio-Temporal DBSCAN Clustering (35m) ]
                               ├── Cluster Found? ──► Deduplicate + Increment Upvote + Compress SLA 15%
                               └── New Cluster?   ──► Create Ticket + Route Dept (PWD/Jal Board/SWM/DISCOM)
                                               │
                                               ▼
                         [ Closed-Loop Proof-of-Fix Vision Audit (/verify-fix) ]
                               └── Photo or Video Audit: Resolves if Defect Count = 0 or Severity Drops > 80%
```

---

## 🚀 Key Algorithmic Innovations & USPs

### 1. Motion Video Evidence & Keyframe Extraction Engine (`services/video_service.py`)
- Citizens and field inspectors can upload **video evidence** (`.mp4`, `.webm`, `.mov`, `.avi`) instead of static photos.
- **Keyframe Extraction:** Automatically extracts keyframes at 2-second intervals using OpenCV.
- **Worst-Case Defect Selection:** Runs the full 3-layer vision pipeline across all keyframes and selects the highest-severity frame as the ticket's visual evidence.
- **Integrated Video Audio Extraction:** Uses `ffmpeg` to isolate the audio track from uploaded videos and transcribes citizen narration using OpenAI Whisper.
- **Interactive Web Player:** Embeds a native video player in the triage inspector with duration and frame metrics.

### 2. 3-Layer Computer Vision Pipeline (`services/vision_service.py`)
Unlike naive single-model detectors, CivicPulse AI uses a tiered defense:
1. **Layer 1 — YOLOv8:** Detects specific localized defect bounding boxes (`Pothole`, `Crack`, `Manhole`, `Garbage`). Supports fine-tuning on RDD2022 (Road Damage Dataset).
2. **Layer 2 — MobileNetV2 CNN Classifier:** Confirms or relabels detections, and applies an **Architectural Line & Scene Filter** to distinguish clean buildings, roads, and skies from genuine civic defects.
3. **Layer 3 — OpenCV Adaptive Structural Analysis:** Canny edge analysis with morphological closing and circularity tests ($4\pi \cdot \text{Area} / \text{Perimeter}^2$) to differentiate circular potholes from linear road cracks, strictly restricted to the road surface.

### 3. False-Positive Elimination & Visual Integrity Guard (`core/severity.py`)
- **Architectural Grid Analysis:** Identifies man-made building facades, windows, and rooflines ($\theta \approx 0^\circ, 90^\circ$ edge dominance) and classifies them as `"Normal / No Defect"` with high confidence.
- **Visual Integrity Guard:** If a user reports a critical hazard (e.g., *"Fallen Tree / Road Hazard"*) but uploads a normal building, selfie, or clean scene, the AI detects 0 visual defects ($C_{\text{conf}} = 0, R_{\text{area}} = 0$).
- **Anti-Hallucination Severity Capping:** The unverified claim is prevented from escalating to an emergency. Severity is anchored at **LOW** ($S \approx 0.11$) and routed to a 120-hour standard desk queue, displaying a green `✓ AI Vision Audit: Stable / No Defect Detected` verification tag.

### 4. Dynamic Multimodal Severity Formula (`core/severity.py`)
Objective mathematical score $S \in [0, 1]$ balancing visual, physical, and environmental risks:

$$S = \text{clip}\left(0.30 \cdot C_{\text{conf}} + 0.35 \cdot R_{\text{area}} + 0.20 \cdot W_{\text{hazard}} + 0.15 \cdot E_{\text{weather}}, 0, 1\right)$$

- $C_{\text{conf}}$: Highest detection confidence from the vision pipeline.
- $R_{\text{area}}$: Cumulative bounding-box-to-image area ratio.
- $W_{\text{hazard}}$: Criticality taxonomy weight (Open Manholes = 1.00, Fallen Trees = 0.90, Waterlogging = 0.85, Potholes = 0.75, Garbage = 0.50).
- $E_{\text{weather}}$: Real-time precipitation hazard fetched live from Open-Meteo at the incident coordinates.

### 5. Spatio-Temporal DBSCAN Deduplication (`core/clustering.py`)
- Uses `scikit-learn` DBSCAN with the Haversine distance metric ($\epsilon = 35\text{ meters}$).
- Incoming grievances within 35 meters of an active ticket are clustered into the parent incident.
- Increases `upvote_count`, merges photographic/video evidence, and **compresses the remaining SLA resolution deadline by 15% per additional citizen confirmation**.

### 6. Closed-Loop "Proof-of-Fix" Verification (`api/v1/tickets/{id}/verify-fix`)
- Contractors or field officers submit an "After" repair photo or video.
- The vision pipeline re-audits the repair zone frame-by-frame.
- Automatically transitions ticket status to `VERIFIED_RESOLVED` only if detected defects drop to zero or calculated severity drops by $> 80\%$.

---

## 📂 Project Structure

```
civicpulse-ai/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── deps.py               # Database session dependency injection
│   │   │   └── v1/
│   │   │       ├── analytics.py      # Department workload & SLA breach risk metrics
│   │   │       ├── reports.py        # Grievance ingestion (photos, videos, audio, GPS)
│   │   │       └── tickets.py        # Ticket queries, GeoJSON feed, and verify-fix audit
│   │   ├── core/
│   │   │   ├── clustering.py         # Haversine DBSCAN 35m deduplication & SLA compression
│   │   │   ├── forensics.py          # 64-bit dHash perceptual hashing & EXIF trust scoring
│   │   │   ├── severity.py           # Multimodal severity formula & visual integrity guard
│   │   │   └── sla_engine.py         # Department routing & dynamic SLA resolution windows
│   │   ├── db/
│   │   │   ├── base.py               # SQLAlchemy declarative base
│   │   │   └── session.py            # SQLite/PostgreSQL engine with schema auto-migration
│   │   ├── models/
│   │   │   └── ticket.py             # Ticket, CitizenReport, and AuditLog ORM models
│   │   ├── schemas/
│   │   │   └── ticket.py             # Pydantic validation schemas (TicketOut, VerifyFix, etc.)
│   │   ├── services/
│   │   │   ├── audio_service.py      # OpenAI Whisper multilingual transcription & NLP
│   │   │   ├── classifier_service.py # MobileNetV2 CNN classifier & zero-shot scene filter
│   │   │   ├── geo_service.py        # OSM Nominatim geocoding & Open-Meteo weather API
│   │   │   ├── video_service.py      # Keyframe extraction, worst-case frame & audio isolation
│   │   │   └── vision_service.py     # 3-Layer triage (YOLOv8 -> MobileNetV2 -> OpenCV)
│   │   ├── config.py                 # Pydantic settings & weights path resolution
│   │   └── main.py                   # FastAPI app, CORS, static media mounting & lifespan
│   └── tests/
│       ├── test_api.py               # End-to-end API tests (image & video workflows)
│       ├── test_clustering.py        # Spatial clustering & SLA acceleration tests
│       └── test_inference.py         # Vision pipeline, dHash, and severity unit tests
├── frontend/
│   └── index.html                    # Real-time command dashboard (Leaflet, Tailwind, Chart.js)
├── scripts/
│   ├── download_and_train.py         # RDD2022 dataset download & YOLOv8 fine-tuning script
│   └── train_classifier.py           # MobileNetV2 civic defect classifier training script
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 🛠️ Quickstart & Local Installation

### Prerequisites
- Python 3.10+ (Python 3.13 recommended)
- `ffmpeg` (optional, for video audio extraction)

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/Mramiit/CivicPulse-AI.git
cd CivicPulse-AI

python -m venv venv
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Start the Server
```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
- **Live Command Dashboard**: Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.
- **Interactive Swagger API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🏋️ Model Training Scripts (Optional)

CivicPulse AI works immediately out-of-the-box in zero-shot mode. To fine-tune custom weights for your municipality:

### 1. Fine-Tune YOLOv8 on Road Damage Dataset (RDD2022)
Downloads the Japan/International RDD2022 dataset (~10k road defect images across potholes and cracks) and trains YOLOv8:
```bash
python scripts/download_and_train.py
```
Outputs trained weights to `weights/civicpulse_yolo.pt`.

### 2. Fine-Tune MobileNetV2 Civic Defect Classifier
Trains the MobileNetV2 classification head on civic defect categories:
```bash
python scripts/train_classifier.py
```
Outputs trained weights to `weights/civicpulse_classifier.pt`.

---

## 🧪 Testing Suite

Execute the test suite covering both image and video workflows:
```bash
$env:PYTHONPATH="."
python backend/tests/test_api.py
```

Tests cover:
- Multimodal severity formula bounds and risk levels.
- Visual integrity guard on clean/architectural images.
- Perceptual `dHash` and EXIF authenticity scoring.
- Haversine DBSCAN 35-meter cluster deduplication and SLA compression.
- End-to-end photo and video grievance report submission.
- Closed-loop Proof-of-Fix verification with photo and video audits.

---

## 📄 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
