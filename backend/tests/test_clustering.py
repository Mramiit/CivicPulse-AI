import pytest
from datetime import datetime, timezone, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.db.base import Base
from backend.app.models.ticket import Ticket, AuditLog
from backend.app.core.clustering import haversine_distance_meters, find_spatial_cluster_match, escalate_crowd_ticket

@pytest.fixture
def test_db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine)
    db = TestingSession()
    yield db
    db.close()

def test_haversine_distance_calculation():
    # Points roughly 30 meters apart in New Delhi
    lat1, lon1 = 28.6139, 77.2090
    lat2, lon2 = 28.6141, 77.2092
    
    dist = haversine_distance_meters(lat1, lon1, lat2, lon2)
    assert 20.0 <= dist <= 40.0

def test_dbscan_clustering_and_crowd_escalation(test_db):
    # Create base ticket
    now = datetime.now(timezone.utc)
    base_ticket = Ticket(
        category="Pothole / Road Damage",
        department="Public Works Department (PWD)",
        severity_score=0.65,
        latitude=28.613939,
        longitude=77.209021,
        status="OPEN",
        upvote_count=1,
        sla_hours=48.0,
        sla_deadline=now + timedelta(hours=48),
        original_image_path="test_img.jpg"
    )
    test_db.add(base_ticket)
    test_db.commit()
    test_db.refresh(base_ticket)

    # 1. Incoming report 20 meters away (should cluster match)
    match = find_spatial_cluster_match(
        test_db,
        latitude=28.614050,
        longitude=77.209100,
        category="Pothole / Road Damage",
        epsilon_meters=35.0
    )
    assert match is not None
    matched_t, dist = match
    assert matched_t.id == base_ticket.id
    assert dist <= 35.0

    # 2. Escalate ticket
    initial_deadline = base_ticket.sla_deadline
    res = escalate_crowd_ticket(
        test_db,
        ticket=matched_t,
        new_description="Water is filling in this pothole too",
        distance_meters=dist
    )
    assert res["upvote_count"] == 2
    assert matched_t.sla_deadline < initial_deadline

    # 3. Report 500 meters away (should NOT cluster match)
    far_match = find_spatial_cluster_match(
        test_db,
        latitude=28.619000,
        longitude=77.215000,
        category="Pothole / Road Damage",
        epsilon_meters=35.0
    )
    assert far_match is None
