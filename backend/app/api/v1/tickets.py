import os
import uuid
import json
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.app.api.deps import get_db
from backend.app.config import settings
from backend.app.models.ticket import Ticket, AuditLog
from backend.app.schemas.ticket import VerifyFixResponse
from backend.app.services.vision_service import run_vision_pipeline
from backend.app.core.severity import calculate_multimodal_severity

router = APIRouter()

def serialize_ticket(t: Ticket) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    deadline = t.sla_deadline
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=timezone.utc)
    
    remaining_seconds = (deadline - now).total_seconds()
    remaining_hours = round(remaining_seconds / 3600.0, 1)

    if t.status == "VERIFIED_RESOLVED":
        sla_status = "RESOLVED"
    elif remaining_seconds <= 0:
        sla_status = "BREACHED"
    elif remaining_hours <= 24.0:
        sla_status = "AT_RISK"
    else:
        sla_status = "ON_TRACK"

    weather_data = {}
    if t.weather_details:
        try:
            weather_data = json.loads(t.weather_details)
        except Exception:
            pass

    detection_data = []
    if t.detection_details:
        try:
            detection_data = json.loads(t.detection_details)
        except Exception:
            pass

    return {
        "id": t.id,
        "tracking_code": t.tracking_code,
        "category": t.category,
        "department": t.department,
        "severity_score": t.severity_score,
        "confidence_score": t.confidence_score,
        "area_ratio": t.area_ratio,
        "hazard_weight": t.hazard_weight,
        "weather_risk": t.weather_risk,
        "latitude": t.latitude,
        "longitude": t.longitude,
        "address": t.address,
        "ward_name": t.ward_name,
        "status": t.status,
        "upvote_count": t.upvote_count,
        "sla_hours": t.sla_hours,
        "sla_deadline": t.sla_deadline.isoformat(),
        "sla_remaining_hours": remaining_hours,
        "sla_status": sla_status,
        "original_image_url": f"/uploads/{t.original_image_path}",
        "annotated_image_url": f"/uploads/{t.annotated_image_path}" if t.annotated_image_path else None,
        "verification_image_url": f"/uploads/{t.verification_image_path}" if t.verification_image_path else None,
        "trust_score": t.trust_score,
        "perceptual_hash": t.perceptual_hash,
        "weather_details": weather_data,
        "detection_details": detection_data,
        "created_at": t.created_at.isoformat(),
        "updated_at": t.updated_at.isoformat(),
        "resolved_at": t.resolved_at.isoformat() if t.resolved_at else None,
        "reports_count": len(t.reports),
        "audit_logs": [
            {
                "id": log.id,
                "action": log.action,
                "details": log.details,
                "timestamp": log.timestamp.isoformat()
            }
            for log in t.audit_logs
        ]
    }


@router.get("/tickets")
def list_tickets(
    status_filter: Optional[str] = Query(None, alias="status"),
    category_filter: Optional[str] = Query(None, alias="category"),
    department_filter: Optional[str] = Query(None, alias="department"),
    geojson: bool = Query(False, description="Return as GeoJSON FeatureCollection"),
    db: Session = Depends(get_db)
):
    """
    Returns list of live civic grievance tickets.
    Can be serialized as standard JSON or GeoJSON FeatureCollection with live SLA countdowns.
    """
    query = db.query(Ticket)
    
    if status_filter:
        query = query.filter(Ticket.status == status_filter.upper())
    if category_filter:
        query = query.filter(Ticket.category == category_filter)
    if department_filter:
        query = query.filter(Ticket.department == department_filter)
        
    tickets = query.order_by(Ticket.created_at.desc()).all()
    serialized = [serialize_ticket(t) for t in tickets]

    if geojson:
        features = []
        for t in serialized:
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [t["longitude"], t["latitude"]]
                },
                "properties": t
            })
        return {
            "type": "FeatureCollection",
            "features": features
        }

    return serialized


@router.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: str, db: Session = Depends(get_db)):
    """Retrieves detailed profile, citizen report history, and audit log for a single ticket."""
    ticket = db.query(Ticket).filter(
        (Ticket.id == ticket_id) | (Ticket.tracking_code == ticket_id.upper())
    ).first()
    
    if not ticket:
        raise HTTPException(status_code=404, detail="Grievance ticket not found.")
        
    return serialize_ticket(ticket)


@router.post("/tickets/{ticket_id}/verify-fix", response_model=VerifyFixResponse)
async def verify_proof_of_fix(
    ticket_id: str,
    image: UploadFile = File(..., description="Post-repair 'After' verification photo"),
    db: Session = Depends(get_db)
):
    """
    Closed-Loop 'Proof-of-Fix' Verification:
    Contractor or citizen submits an 'After' photo.
    Runs vision pipeline: only transitions ticket to VERIFIED_RESOLVED if
    defect bounding box count drops to zero OR severity drops by > 80%.
    """
    ticket = db.query(Ticket).filter(
        (Ticket.id == ticket_id) | (Ticket.tracking_code == ticket_id.upper())
    ).first()

    if not ticket:
        raise HTTPException(status_code=404, detail="Grievance ticket not found.")

    if ticket.status == "VERIFIED_RESOLVED":
        return VerifyFixResponse(
            ticket_id=ticket.id,
            tracking_code=ticket.tracking_code,
            status=ticket.status,
            verified=True,
            initial_severity=ticket.severity_score,
            after_severity=0.0,
            reduction_percentage=100.0,
            detected_defects_after=0,
            message="Ticket is already verified and marked RESOLVED."
        )

    # Save verification image
    file_id = uuid.uuid4().hex
    img_ext = os.path.splitext(image.filename)[1] or ".jpg"
    after_img_name = f"after_{file_id}{img_ext}"
    after_img_path = os.path.join(settings.UPLOAD_DIR, after_img_name)

    content = await image.read()
    with open(after_img_path, "wb") as f:
        f.write(content)

    annotated_after_name = f"annotated_after_{file_id}.jpg"
    annotated_after_path = os.path.join(settings.UPLOAD_DIR, annotated_after_name)

    # Run vision pipeline on After image
    detections, c_conf, r_area, _ = run_vision_pipeline(after_img_path, annotated_after_path)
    
    # Calculate After severity
    after_severity, _ = calculate_multimodal_severity(
        c_conf=c_conf if detections else 0.0,
        r_area=r_area if detections else 0.0,
        category=ticket.category,
        e_weather=0.0  # Normalized weather for repair inspection
    )

    initial_severity = max(0.01, ticket.severity_score)
    severity_reduction = (initial_severity - after_severity) / initial_severity
    reduction_percentage = round(max(0.0, severity_reduction * 100.0), 1)

    defect_count = len(detections)
    # Verification condition: zero bounding boxes OR severity drops by > 80%
    is_verified = (defect_count == 0) or (reduction_percentage >= 80.0)

    now = datetime.now(timezone.utc)
    ticket.verification_image_path = annotated_after_name

    if is_verified:
        ticket.status = "VERIFIED_RESOLVED"
        ticket.resolved_at = now
        log_action = "PROOF_OF_FIX_ACCEPTED"
        message = (
            f"Verification SUCCESS! Defect count: {defect_count}. "
            f"Severity reduced by {reduction_percentage}%. Ticket transitioned to VERIFIED_RESOLVED."
        )
    else:
        log_action = "PROOF_OF_FIX_REJECTED"
        message = (
            f"Verification REJECTED. Defect still detectable ({defect_count} anomalies). "
            f"Severity reduction was only {reduction_percentage}% (Required: >80%)."
        )

    audit = AuditLog(
        ticket_id=ticket.id,
        action=log_action,
        details=message,
        timestamp=now
    )
    db.add(audit)
    db.commit()
    db.refresh(ticket)

    return VerifyFixResponse(
        ticket_id=ticket.id,
        tracking_code=ticket.tracking_code,
        status=ticket.status,
        verified=is_verified,
        initial_severity=ticket.severity_score,
        after_severity=after_severity,
        reduction_percentage=reduction_percentage,
        detected_defects_after=defect_count,
        annotated_after_image_url=f"/uploads/{annotated_after_name}",
        message=message
    )
