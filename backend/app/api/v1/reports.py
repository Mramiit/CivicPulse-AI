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
from backend.app.services.video_service import is_video_file, analyze_video, extract_audio_from_video

router = APIRouter()


@router.post("/reports", status_code=status.HTTP_201_CREATED)
async def submit_citizen_report(
    latitude: float = Form(..., description="Live GPS Latitude"),
    longitude: float = Form(..., description="Live GPS Longitude"),
    description: Optional[str] = Form(None, description="Citizen grievance description"),
    category_hint: Optional[str] = Form(None, description="Defect category hint"),
    media: Optional[UploadFile] = File(None, description="Defect photo OR video upload (image/*, video/mp4, video/webm, etc.)"),
    image: Optional[UploadFile] = File(None, description="Backwards-compatible alias for photo/video upload"),
    audio: Optional[UploadFile] = File(None, description="Citizen voice dispatch (optional, separate audio file)"),
    db: Session = Depends(get_db)
):
    """
    Ingests multimodal citizen grievance report with IMAGE or VIDEO proof:

    Image upload:
      1. Perceptual hashing (dHash) & EXIF anti-spoofing
      2. 3-layer vision pipeline (YOLO → CNN classifier → OpenCV)

    Video upload:
      1. Keyframe extraction (every 2s, up to 8 frames)
      2. 3-layer vision pipeline on EACH frame → pick worst-case
      3. Audio track extraction (ffmpeg) → Whisper transcription
      4. Best frame used as representative image for ticket

    Common pipeline (both):
      3. Live OpenStreetMap Nominatim reverse geocoding
      4. Live Open-Meteo environmental weather risk
      5. Multimodal mathematical severity calculation (with visual verification guard)
      6. Spatio-temporal DBSCAN deduplication (35m radius)
    """
    upload = media or image
    if upload is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="A defect photo or video file is required (pass 'media' or 'image')."
        )

    file_id = uuid.uuid4().hex
    file_ext = os.path.splitext(upload.filename or "")[1].lower() or ".jpg"
    is_video = is_video_file(upload.filename or "")

    # ── Save uploaded media ───────────────────────────────────────────────────
    media_content = await upload.read()

    raw_img_name = ""
    raw_img_path = ""
    annotated_img_name = ""
    annotated_img_path = ""
    video_name = ""
    video_path_on_disk = ""
    video_meta = {}
    media_type = "video" if is_video else "image"

    if is_video:
        # ── VIDEO PATH ────────────────────────────────────────────────────────
        video_name = f"video_{file_id}{file_ext}"
        video_path_on_disk = os.path.join(settings.UPLOAD_DIR, video_name)
        with open(video_path_on_disk, "wb") as f:
            f.write(media_content)

        # Run full video analysis pipeline (keyframe extraction → per-frame vision)
        detections, c_conf, r_area, best_frame_path, annotated_path, video_meta = \
            analyze_video(video_path_on_disk, settings.UPLOAD_DIR, max_frames=8)

        # Use the best frame as the representative image
        if best_frame_path and os.path.exists(best_frame_path):
            raw_img_name = os.path.basename(best_frame_path)
            raw_img_path = best_frame_path
        else:
            raw_img_name = f"report_{file_id}.jpg"
            raw_img_path = os.path.join(settings.UPLOAD_DIR, raw_img_name)

        if annotated_path and os.path.exists(annotated_path):
            annotated_img_name = os.path.basename(annotated_path)
        else:
            annotated_img_name = ""

        # Extract audio from video for Whisper transcription (if no separate audio uploaded)
        video_audio_path = None
        if audio is None or not (audio.filename or "").strip():
            video_audio_path = extract_audio_from_video(video_path_on_disk, settings.UPLOAD_DIR)

    else:
        # ── IMAGE PATH ────────────────────────────────────────────────────────
        raw_img_name = f"report_{file_id}{file_ext}"
        raw_img_path = os.path.join(settings.UPLOAD_DIR, raw_img_name)
        with open(raw_img_path, "wb") as f:
            f.write(media_content)

        annotated_img_name = f"annotated_{file_id}.jpg"
        annotated_img_path = os.path.join(settings.UPLOAD_DIR, annotated_img_name)

        # Run 3-layer vision pipeline
        detections, c_conf, r_area, preview_b64 = run_vision_pipeline(raw_img_path, annotated_img_path)

    # ── Perceptual hash & forensics ──────────────────────────────────────────
    p_hash = compute_dhash(raw_img_path) if os.path.exists(raw_img_path) else ""
    trust_score, exif_meta = inspect_image_forensics(raw_img_path) if os.path.exists(raw_img_path) else (0.75, {})

    # ── Audio processing ──────────────────────────────────────────────────────
    audio_path = None
    transcript = ""
    urgency_keywords = []

    if audio is not None and (audio.filename or "").strip():
        audio_ext = os.path.splitext(audio.filename)[1] or ".webm"
        audio_name = f"audio_{file_id}{audio_ext}"
        audio_path = os.path.join(settings.UPLOAD_DIR, audio_name)
        audio_content = await audio.read()
        with open(audio_path, "wb") as f:
            f.write(audio_content)
        transcript, urgency_keywords, audio_meta = transcribe_and_analyze_audio(audio_path)

    elif is_video and video_audio_path:
        audio_path = video_audio_path
        transcript, urgency_keywords, audio_meta = transcribe_and_analyze_audio(audio_path)

    # ── Category resolution ───────────────────────────────────────────────────
    detected_labels = [d["label"].lower() for d in detections]
    combined_notes = f"{description or ''} {transcript}".lower()

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

    # ── Live geocoding & weather ──────────────────────────────────────────────
    address_str, ward_str = await reverse_geocode_nominatim(latitude, longitude)
    e_weather, weather_details = await fetch_live_weather_risk(latitude, longitude)

    # ── Department routing & multimodal severity ──────────────────────────────
    department = route_department(category, combined_notes)
    severity_score, severity_breakdown = calculate_multimodal_severity(c_conf, r_area, category, e_weather)

    # ── DBSCAN deduplication ──────────────────────────────────────────────────
    cluster_match = find_spatial_cluster_match(
        db,
        latitude=latitude,
        longitude=longitude,
        category=category,
        epsilon_meters=settings.DBSCAN_EPSILON_METERS
    )

    if cluster_match is not None:
        matched_ticket, dist_meters = cluster_match

        crowd_report = CitizenReport(
            ticket_id=matched_ticket.id,
            latitude=latitude,
            longitude=longitude,
            user_description=description,
            transcribed_text=transcript,
            urgency_keywords=json.dumps(urgency_keywords),
            image_path=raw_img_name,
            audio_path=os.path.basename(audio_path) if audio_path else None,
            video_path=video_name if is_video else None,
            perceptual_hash=p_hash
        )
        db.add(crowd_report)

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
            "media_type": media_type,
            "upvote_count": matched_ticket.upvote_count,
            "severity_score": matched_ticket.severity_score,
            "sla_deadline": matched_ticket.sla_deadline.isoformat(),
            "annotated_image_url": f"/uploads/{matched_ticket.annotated_image_path}" if matched_ticket.annotated_image_path else None,
            "cluster_distance_meters": dist_meters
        }

    # ── Create new ticket ─────────────────────────────────────────────────────
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
        video_path=video_name if is_video else None,
        media_type=media_type,
        trust_score=trust_score,
        perceptual_hash=p_hash,
        weather_details=json.dumps(weather_details),
        detection_details=json.dumps(detections),
        video_analysis_details=json.dumps(video_meta) if video_meta else None,
    )
    db.add(new_ticket)
    db.flush()

    first_report = CitizenReport(
        ticket_id=new_ticket.id,
        latitude=latitude,
        longitude=longitude,
        user_description=description,
        transcribed_text=transcript,
        urgency_keywords=json.dumps(urgency_keywords),
        image_path=raw_img_name,
        audio_path=os.path.basename(audio_path) if audio_path else None,
        video_path=video_name if is_video else None,
        perceptual_hash=p_hash
    )
    db.add(first_report)

    media_detail = f"Video ({video_meta.get('duration_seconds', 0)}s, {video_meta.get('frames_analyzed', 0)} frames analyzed)" if is_video else "Image"
    visual_audit_note = " [NOTE: Visual AI detected no defect in uploaded media — routed at Low priority desk queue.]" if (c_conf == 0.0 and r_area == 0.0) else ""

    audit = AuditLog(
        ticket_id=new_ticket.id,
        action="TICKET_CREATED",
        details=(
            f"Grievance initialized via {media_detail}. Category: {category}. Dept: {department}. "
            f"Severity: {severity_score} ({severity_breakdown['risk_level']}). "
            f"Initial SLA: {sla_hours}h. Trust: {trust_score}.{visual_audit_note}"
        ),
        timestamp=datetime.now(timezone.utc)
    )
    db.add(audit)
    db.commit()
    db.refresh(new_ticket)

    response = {
        "status": "CREATED",
        "message": f"New grievance ticket #{new_ticket.tracking_code} registered successfully.",
        "ticket_id": new_ticket.id,
        "tracking_code": new_ticket.tracking_code,
        "category": new_ticket.category,
        "department": new_ticket.department,
        "media_type": media_type,
        "severity_score": new_ticket.severity_score,
        "severity_breakdown": severity_breakdown,
        "address": new_ticket.address,
        "ward_name": new_ticket.ward_name,
        "sla_hours": new_ticket.sla_hours,
        "sla_deadline": new_ticket.sla_deadline.isoformat(),
        "annotated_image_url": f"/uploads/{new_ticket.annotated_image_path}" if new_ticket.annotated_image_path else None,
        "weather_info": weather_details,
        "trust_score": new_ticket.trust_score,
        "visual_status": "DEFECT_DETECTED" if c_conf > 0 else "NO_DEFECT_DETECTED"
    }

    if is_video:
        response["video_url"] = f"/uploads/{video_name}"
        response["video_analysis"] = {
            "duration_seconds": video_meta.get("duration_seconds", 0),
            "frames_analyzed": video_meta.get("frames_analyzed", 0),
            "best_frame_index": video_meta.get("best_frame_index", 0),
            "audio_transcribed": bool(transcript),
        }

    return response
