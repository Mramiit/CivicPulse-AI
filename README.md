# CivicPulse AI: Autonomous Multimodal Civic Grievance & Infrastructure Triage Engine

[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![YOLOv8](https://img.shields.io/badge/Vision-Ultralytics%20YOLOv8-FF7F00.svg)](https://github.com/ultralytics/ultralytics)
[![Whisper](https://img.shields.io/badge/Audio-OpenAI%20Whisper-00A67E.svg)](https://github.com/openai/whisper)
[![Leaflet](https://img.shields.io/badge/Geospatial-Leaflet.js-199900.svg)](https://leafletjs.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

CivicPulse AI is an enterprise-grade, autonomous civic grievance triage system. It continuously ingests multimodal citizen reports (high-resolution photographs, Web Audio API voice dispatches, and live HTML5 GPS coordinates), dynamically analyzes defects using Computer Vision, enriches incident context using live OpenStreetMap and Open-Meteo REST APIs, and dispatches actionable tickets to municipal departments with dynamic SLA deadlines.

---

## 🏛️ System Architecture

```
[ Citizen Smartphone / Web Dashboard ]
       │
       ├── 1. Live HTML5 Geolocation (navigator.geolocation) & Map Click
       ├── 2. Defect Photograph (Camera / File Upload)
       └── 3. Web Audio API Voice Recording
                 │
                 ▼
     [ FastAPI Ingestion Gateway (/api/v1/reports) ]
                 │
  ┌──────────────┼──────────────┬──────────────┐
  ▼              ▼              ▼              ▼
[ EXIF & dHash ] [ Live OSM ]  [ Open-Meteo ]  [ Multimodal ]
Perceptual Hash  Nominatim      Live Weather    AI Engines
Anti-Spoofing    Reverse-Geo    Precipitation   YOLOv8 + OpenCV
  │              │              │               Whisper Audio
  └──────────────┼──────────────┴──────────────┘
                 ▼
  [ Multimodal Severity Formulation Engine ]
     S = clip(0.30·C_conf + 0.35·R_area + 0.20·W_hazard + 0.15·E_weather, 0, 1)
                 │
                 ▼
  [ Spatio-Temporal DBSCAN Deduplication (Haversine 35m) ]
       ├── Cluster Match Found? ──► Increment Upvotes + 15% SLA Dynamic Acceleration
       └── New Cluster? ──────────► Create Ticket + Route Dept (PWD / Jal Board / SWM / DISCOM)
                 │
                 ▼
  [ Closed-Loop Proof-of-Fix Vision Verification (/verify-fix) ]
       └── "After" Photo Audit: Resolves if Defect Count = 0 or Severity Drops > 80%
```

---

## 🚀 4 Core Algorithmic USPs

### 1. Multimodal Defect + Context Severity Engine (`core/severity.py`)
Computes an objective mathematical severity score $S \in [0, 1]$ combining visual defect geometry, hazard criticality, and live environmental risks:

$$S = \text{clip}\left(0.30 \cdot C_{\text{conf}} + 0.35 \cdot R_{\text{area}} + 0.20 \cdot W_{\text{hazard}} + 0.15 \cdot E_{\text{weather}}, 0, 1\right)$$

- $C_{\text{conf}}$: Visual model detection confidence.
- $R_{\text{area}}$: Bounding-box-to-image area ratio.
- $W_{\text{hazard}}$: Defect criticality taxonomy weight (e.g., Open Manholes = 1.00, Waterlogging = 0.85, Potholes = 0.75).
- $E_{\text{weather}}$: Real-time precipitation hazard fetched live from Open-Meteo at the incident coordinates.

### 2. EXIF & Perceptual Anti-Spoofing Guardrail (`core/forensics.py`)
- Computes a 64-bit difference hash (`dHash`) for every photograph.
- Uses Hamming distance to detect duplicate image spam across different GPS coordinates, preventing fraudulent bounty claims.
- Assesses EXIF metadata (camera model, timestamps) to generate a verifiable `trust_score`.

### 3. Spatio-Temporal DBSCAN Deduplication & Crowd Escalation (`core/clustering.py`)
- Executes `sklearn.cluster.DBSCAN` with the Haversine metric ($\epsilon = 35\text{ meters}$, `min_samples=1`).
- Incoming reports within 35 meters of an active ticket are deduplicated into the existing incident.
- Rather than spamming municipal workers with duplicate tickets, it increases the ticket's `upvote_count`, merges evidence, and **compresses remaining SLA resolution time by 15% per additional citizen confirmation**.

### 4. Closed-Loop "Proof-of-Fix" Verification (`api/v1/tickets/{id}/verify-fix`)
- Contractors or citizens submit an "After" repair photo.
- The computer vision engine re-audits the repair zone.
- Transitions status to `VERIFIED_RESOLVED` only if detected defect bounding box count drops to zero or calculated severity drops by $> 80\%$.

---

## 🛠️ Quickstart & Local Installation

### Prerequisites
- Python 3.10+
- (Optional) Docker & Docker Compose

### 1. Clone & Set Up Environment
```bash
git clone <repo-url> civicpulse-ai
cd civicpulse-ai

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Application
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
- **Live Command Dashboard**: Open `http://localhost:8000` in your browser.
- **Interactive OpenAPI Documentation**: `http://localhost:8000/docs`

---

## 🐳 Docker Deployment
```bash
docker-compose up --build -d
```

---

## 🧪 Testing Suite
Execute unit and integration tests:
```bash
pytest backend/tests/
```
Tests cover:
- Multimodal severity formula bounds and risk levels.
- Perceptual `dHash` and EXIF authenticity scoring.
- Haversine DBSCAN 35-meter cluster deduplication and SLA compression.
- End-to-end report submission, listing, and Proof-of-Fix validation.
