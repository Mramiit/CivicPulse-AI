from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime, timezone, timedelta
import numpy as np
from sklearn.cluster import DBSCAN
from sqlalchemy.orm import Session

from backend.app.models.ticket import Ticket, AuditLog
from backend.app.config import settings

EARTH_RADIUS_METERS = 6371000.0

def haversine_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes exact Haversine distance in meters between two GPS coordinates."""
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return float(EARTH_RADIUS_METERS * c)

def find_spatial_cluster_match(
    db: Session,
    latitude: float,
    longitude: float,
    category: str,
    epsilon_meters: float = 35.0
) -> Optional[Tuple[Ticket, float]]:
    """
    Finds if an active, unresolved ticket within epsilon_meters exists for the same category
    using scikit-learn DBSCAN with Haversine distance metric.
    
    Returns:
        (matched_ticket, distance_meters) or None
    """
    # Query all active tickets in the same category
    active_tickets: List[Ticket] = db.query(Ticket).filter(
        Ticket.category == category,
        Ticket.status.in_(["OPEN", "IN_PROGRESS"])
    ).all()

    if not active_tickets:
        return None

    # Prepare coordinates in radians for DBSCAN [lat, lon]
    existing_coords = np.array([
        [np.radians(t.latitude), np.radians(t.longitude)]
        for t in active_tickets
    ])
    
    incoming_coord = np.array([[np.radians(latitude), np.radians(longitude)]])
    
    # Combined coordinates with incoming point as index 0
    all_coords = np.vstack([incoming_coord, existing_coords])
    
    eps_radians = epsilon_meters / EARTH_RADIUS_METERS

    # Run DBSCAN with Haversine metric
    dbscan = DBSCAN(eps=eps_radians, min_samples=1, metric="haversine")
    labels = dbscan.fit_predict(all_coords)

    target_label = labels[0]
    
    # Check if any existing ticket shares the same cluster label
    for idx, ticket in enumerate(active_tickets):
        # Index in all_coords is idx + 1
        if labels[idx + 1] == target_label:
            dist = haversine_distance_meters(latitude, longitude, ticket.latitude, ticket.longitude)
            if dist <= epsilon_meters:
                return ticket, round(dist, 1)

    return None

def escalate_crowd_ticket(
    db: Session,
    ticket: Ticket,
    new_description: Optional[str] = None,
    new_transcript: Optional[str] = None,
    distance_meters: float = 0.0
) -> Dict[str, Any]:
    """
    Handles crowd deduplication escalation:
    - Increments upvote_count
    - Dynamically reduces remaining SLA deadline by 15%
    - Logs audit record
    """
    ticket.upvote_count += 1
    now = datetime.now(timezone.utc)

    # Calculate remaining SLA time and compress it by 15%
    old_deadline = ticket.sla_deadline
    if old_deadline.tzinfo is None:
        old_deadline = old_deadline.replace(tzinfo=timezone.utc)
        
    remaining_time = old_deadline - now
    if remaining_time.total_seconds() > 0:
        compressed_seconds = remaining_time.total_seconds() * 0.85
        ticket.sla_deadline = now + timedelta(seconds=compressed_seconds)
        sla_hours_reduced = (remaining_time.total_seconds() - compressed_seconds) / 3600.0
    else:
        sla_hours_reduced = 0.0

    # Create Audit Log
    escalation_detail = (
        f"Crowdsourced confirmation added (Upvotes: {ticket.upvote_count}). "
        f"Location delta: {distance_meters}m. "
        f"SLA deadline accelerated by 15% (-{round(sla_hours_reduced, 1)}h)."
    )
    if new_description:
        escalation_detail += f" Additional citizen note: '{new_description}'"
    if new_transcript:
        escalation_detail += f" Voice transcript merged: '{new_transcript}'"

    audit = AuditLog(
        ticket_id=ticket.id,
        action="CROWD_ESCALATION",
        details=escalation_detail,
        timestamp=now
    )
    db.add(audit)
    db.commit()
    db.refresh(ticket)

    return {
        "action": "ESCALATED",
        "ticket_id": ticket.id,
        "tracking_code": ticket.tracking_code,
        "upvote_count": ticket.upvote_count,
        "new_sla_deadline": ticket.sla_deadline.isoformat(),
        "distance_meters": distance_meters
    }
