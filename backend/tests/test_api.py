import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

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

def test_health_check_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "online"

def test_submit_report_and_verify_fix_workflow():
    img_bytes = create_test_image_bytes()
    
    # 1. Submit citizen report
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

    # 4. Proof of fix verification
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
