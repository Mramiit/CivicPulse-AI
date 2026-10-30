import os
import uuid
import json
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.api.deps import get_db
from backend.app.config import settings
from backend.app.models.ticket import Ticket, CitizenReport, AuditLog
from backend.app.core.forensics import compute_dhash, inspect_image_forensics
from backend.app.core.severity import calculate_multimodal_severity
from backend.app.core.clustering import find_spatial_cluster_match, escalate_crowd_ticket
from backend.app.core.sla_engine import route_department, calculate_initial_sla
from backend.app.services.geo_service import reverse_geocode_nominatim, fetch_live_weather_risk
from backend.app.services.vision_service import run_vision_pipeline
from backend.app.services.audio_service import transcribe_and_analyze_audio

router = APIRouter()

@router.post("/reports", status_code=status.HTTP_201_CREATED)
async def submit_citizen_report(
    latitude: float = Form(..., description="Live GPS Latitude"),
    longitude: float = Form(..., description="Live GPS Longitude"),
    description: Optional[str] = Form(None, description="Citizen grievance description"),
    category_hint: Optional[str] = Form(None, description="Defect category hint"),
    image: UploadFile = File(..., description="Defect photo upload"),
    audio: Optional[UploadFile] = File(None, description="Citizen voice dispatch"),
    db: Session = Depends(get_db)
):
    """
    Ingests multimodal citizen grievance report:
    1. Perceptual hashing (dHash) & EXIF anti-spoofing inspection.
    2. Live OpenStreetMap Nominatim reverse geocoding.
    3. Live Open-Meteo environmental weather risk amplification.
    4. YOLOv8 + OpenCV contour vision pipeline.
    5. Whisper audio transcription and urgency analysis.
    6. Multimodal mathematical severity calculation.
    7. Spatio-temporal DBSCAN deduplication (35m radius) & dynamic crowd escalation.
    """
    file_id = uuid.uuid4().hex
    
    # 1. Save uploaded image
    img_ext = os.path.splitext(image.filename)[1] or ".jpg"
    raw_img_name = f"report_{file_id}{img_ext}"
    raw_img_path = os.path.join(settings.UPLOAD_DIR, raw_img_name)
    
    content = await image.read()
    with open(raw_img_path, "wb") as f:
        f.write(content)

    # Compute perceptual hash & forensics
    p_hash = compute_dhash(raw_img_path)
    trust_score, exif_meta = inspect_image_forensics(raw_img_path)

    # 2. Save and process optional audio
    audio_path = None
    transcript = ""
    urgency_keywords = []
    
    if audio is not None and audio.filename:
        audio_ext = os.path.splitext(audio.filename)[1] or ".webm"
        audio_name = f"audio_{file_id}{audio_ext}"
        audio_path = os.path.join(settings.UPLOAD_DIR, audio_name)
        audio_content = await audio.read()
        with open(audio_path, "wb") as f:
            f.write(audio_content)
        
        transcript, urgency_keywords, audio_meta = transcribe_and_analyze_audio(audio_path)

    # 3. Vision Detection Pipeline
    annotated_img_name = f"annotated_{file_id}.jpg"
    annotated_img_path = os.path.join(settings.UPLOAD_DIR, annotated_img_name)
    
    detections, c_conf, r_area, preview_b64 = run_vision_pipeline(raw_img_path, annotated_img_path)

    # 4. Determine Defect Category
    detected_labels = [d["label"].lower() for d in detections]
    combined_notes = f"{description or ''} {transcript}".lower()
    
    # Category resolution priority
    if category_hint and len(category_hint.strip()) > 0:
        category = category_hint.strip()
    elif any("manhole" in lbl for lbl in detected_labels) or "manhole" in combined_notes:
        category = "Open Manhole / Drain"
    elif any("tree" in lbl for lbl in detected_labels) or "tree" in combined_notes:
        category = "Fallen Tree / Road Hazard"
    elif "water" in combined_notes or "flood" in combined_notes:
        category = "Waterlogging & Drainage"
    elif any("garbage" in lbl or "trash" in lbl for lbl in detected_labels) or "garbage" in combined_notes:
        category = "Solid Waste & Garbage"
    elif "wire" in combined_notes or "electric" in combined_notes or "light" in combined_notes:
        category = "Streetlight / Electrical Hazard"
    else:
        category = "Pothole / Road Damage"

    # 5. Live Reverse Geocoding (Nominatim) & Weather Hazard (Open-Meteo)
    address_str, ward_str = await reverse_geocode_nominatim(latitude, longitude)
    e_weather, weather_details = await fetch_live_weather_risk(latitude, longitude)

    # 6. Route Department & Calculate Dynamic Severity
    department = route_department(category, combined_notes)
    severity_score, severity_breakdown = calculate_multimodal_severity(c_conf, r_area, category, e_weather)

    # 7. DBSCAN Deduplication (35-meter cluster radius)
    cluster_match = find_spatial_cluster_match(
        db,
        latitude=latitude,
        longitude=longitude,
        category=category,
        epsilon_meters=settings.DBSCAN_EPSILON_METERS
    )

    if cluster_match is not None:
        matched_ticket, dist_meters = cluster_match
        
        # Log citizen report attached to existing ticket
        crowd_report = CitizenReport(
            ticket_id=matched_ticket.id,
            latitude=latitude,
            longitude=longitude,
            user_description=description,
            transcribed_text=transcript,
            urgency_keywords=json.dumps(urgency_keywords),
            image_path=raw_img_name,
            audio_path=os.path.basename(audio_path) if audio_path else None,
            perceptual_hash=p_hash
        )
        db.add(crowd_report)
        
        # Crowd escalation & 15% SLA compression
        escalate_info = escalate_crowd_ticket(
            db,
            ticket=matched_ticket,
            new_description=description,
            new_transcript=transcript,
            distance_meters=dist_meters
        )

        return {
            "status": "CROWD_ESCALATED",
            "message": f"Deduplicated! Clustered with active ticket #{matched_ticket.tracking_code} ({dist_meters}m away). Crowd confirmation recorded and SLA compressed by 15%.",
            "ticket_id": matched_ticket.id,
            "tracking_code": matched_ticket.tracking_code,
            "category": matched_ticket.category,
            "upvote_count": matched_ticket.upvote_count,
            "severity_score": matched_ticket.severity_score,
            "sla_deadline": matched_ticket.sla_deadline.isoformat(),
            "annotated_image_url": f"/uploads/{matched_ticket.annotated_image_path}" if matched_ticket.annotated_image_path else None,
            "cluster_distance_meters": dist_meters
        }

    # 8. Create New Ticket
    sla_hours, sla_deadline = calculate_initial_sla(severity_score)

    new_ticket = Ticket(
        category=category,
        department=department,
        severity_score=severity_score,
        confidence_score=c_conf,
        area_ratio=r_area,
        hazard_weight=severity_breakdown["hazard_weight_raw"],
        weather_risk=e_weather,
        latitude=latitude,
        longitude=longitude,
        address=address_str,
        ward_name=ward_str,
        status="OPEN",
        upvote_count=1,
        sla_hours=sla_hours,
        sla_deadline=sla_deadline,
        original_image_path=raw_img_name,
        annotated_image_path=annotated_img_name,
        trust_score=trust_score,
        perceptual_hash=p_hash,
        weather_details=json.dumps(weather_details),
        detection_details=json.dumps(detections)
    )
    db.add(new_ticket)
    db.flush()

    # Create associated Citizen Report
    first_report = CitizenReport(
        ticket_id=new_ticket.id,
        latitude=latitude,
        longitude=longitude,
        user_description=description,
        transcribed_text=transcript,
        urgency_keywords=json.dumps(urgency_keywords),
        image_path=raw_img_name,
        audio_path=os.path.basename(audio_path) if audio_path else None,
        perceptual_hash=p_hash
    )
    db.add(first_report)

    # Initial Audit Log
    audit = AuditLog(
        ticket_id=new_ticket.id,
        action="TICKET_CREATED",
        details=(
            f"Grievance initialized. Category: {category}. Dept: {department}. "
            f"Severity: {severity_score} ({severity_breakdown['risk_level']}). "
            f"Initial SLA: {sla_hours}h. Trust: {trust_score}."
        ),
        timestamp=datetime.now(timezone.utc)
    )
    db.add(audit)
    db.commit()
    db.refresh(new_ticket)

    return {
        "status": "CREATED",
        "message": f"New grievance ticket #{new_ticket.tracking_code} registered successfully.",
        "ticket_id": new_ticket.id,
        "tracking_code": new_ticket.tracking_code,
        "category": new_ticket.category,
        "department": new_ticket.department,
        "severity_score": new_ticket.severity_score,
        "severity_breakdown": severity_breakdown,
        "address": new_ticket.address,
        "ward_name": new_ticket.ward_name,
        "sla_hours": new_ticket.sla_hours,
        "sla_deadline": new_ticket.sla_deadline.isoformat(),
        "annotated_image_url": f"/uploads/{new_ticket.annotated_image_path}",
        "weather_info": weather_details,
        "trust_score": new_ticket.trust_score
    }
