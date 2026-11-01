import io
import os
import tempfile
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import pytest
except ImportError:
    pytest = None

from backend.app.main import app
from backend.app.db.session import engine
from backend.app.db.base import Base
import backend.app.models.ticket

# Ensure database tables exist
Base.metadata.create_all(bind=engine)

client = TestClient(app)

def create_test_image_bytes():
    buf = io.BytesIO()
    img = Image.new("RGB", (320, 240), color=(120, 120, 120))
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf

def create_test_video_bytes():
    """Generates a small valid MP4 video in-memory for testing."""
    if not HAS_CV2:
        return io.BytesIO(b"dummy_video_bytes")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name

    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, 10.0, (160, 120))
        for _ in range(25):
            frame = np.full((120, 160, 3), 110, dtype=np.uint8)
            # Add a dark patch representing a defect
            frame[40:80, 50:110] = 30
            out.write(frame)
        out.release()

        with open(path, "rb") as vf:
            data = vf.read()
        return io.BytesIO(data)
    finally:
        if os.path.exists(path):
            os.remove(path)

def test_health_check_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "online"

def test_submit_report_and_verify_fix_workflow():
    img_bytes = create_test_image_bytes()
    
    # 1. Submit citizen report using legacy 'image' field
    response = client.post(
        "/api/v1/reports",
        data={
            "latitude": 28.6139,
            "longitude": 77.2090,
            "description": "Deep dangerous crater in main road intersection",
            "category_hint": "Pothole / Road Damage"
        },
        files={
            "image": ("pothole.jpg", img_bytes, "image/jpeg")
        }
    )
    assert response.status_code == 201
    data = response.json()
    assert "ticket_id" in data
    assert "tracking_code" in data
    assert data["media_type"] == "image"
    ticket_id = data["ticket_id"]

    # 2. Query tickets list
    list_res = client.get("/api/v1/tickets")
    assert list_res.status_code == 200
    tickets = list_res.json()
    assert any(t["id"] == ticket_id for t in tickets)

    # 3. Query analytics
    analytics_res = client.get("/api/v1/analytics")
    assert analytics_res.status_code == 200
    analytics = analytics_res.json()
    assert analytics["total_tickets"] >= 1
    assert "sla_breach_risk" in analytics

    # 4. Proof of fix verification using legacy 'image' field
    after_bytes = create_test_image_bytes()
    fix_res = client.post(
        f"/api/v1/tickets/{ticket_id}/verify-fix",
        files={
            "image": ("repaired_road.jpg", after_bytes, "image/jpeg")
        }
    )
    assert fix_res.status_code == 200
    fix_data = fix_res.json()
    assert "verified" in fix_data
    assert "reduction_percentage" in fix_data


def test_video_report_and_verify_fix_workflow():
    if not HAS_CV2:
        return

    vid_bytes = create_test_video_bytes()

    # 1. Submit citizen report with VIDEO proof using new 'media' field
    response = client.post(
        "/api/v1/reports",
        data={
            "latitude": 28.7041,
            "longitude": 77.1025,
            "description": "Severe road cavity captured in motion video",
            "category_hint": "Pothole / Road Damage"
        },
        files={
            "media": ("defect_proof.mp4", vid_bytes, "video/mp4")
        }
    )
    assert response.status_code == 201
    data = response.json()
    assert "ticket_id" in data
    assert "tracking_code" in data
    assert data["media_type"] == "video"
    assert "video_url" in data
    assert "video_analysis" in data
    assert data["video_analysis"]["frames_analyzed"] >= 1
    ticket_id = data["ticket_id"]

    # 2. Verify ticket serialization has video fields
    ticket_res = client.get(f"/api/v1/tickets/{ticket_id}")
    assert ticket_res.status_code == 200
    ticket_info = ticket_res.json()
    assert ticket_info["media_type"] == "video"
    assert ticket_info["video_url"] is not None
    assert ticket_info["video_analysis"] is not None

    # 3. Verify fix using VIDEO proof
    after_vid_bytes = create_test_video_bytes()
    fix_res = client.post(
        f"/api/v1/tickets/{ticket_id}/verify-fix",
        files={
            "media": ("repaired_pan.mp4", after_vid_bytes, "video/mp4")
        }
    )
    assert fix_res.status_code == 200
    fix_data = fix_res.json()
    assert "verified" in fix_data
    assert "reduction_percentage" in fix_data
    assert fix_data["verification_video_url"] is not None


if __name__ == "__main__":
    test_health_check_endpoint()
    print("[PASS] test_health_check_endpoint")
    test_submit_report_and_verify_fix_workflow()
    print("[PASS] test_submit_report_and_verify_fix_workflow")
    test_video_report_and_verify_fix_workflow()
    print("[PASS] test_video_report_and_verify_fix_workflow")
    print("ALL API TESTS PASSED!")
