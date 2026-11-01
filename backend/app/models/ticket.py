import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Integer, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from backend.app.db.base import Base

def generate_uuid():
    return str(uuid.uuid4())

def generate_tracking_code():
    return f"CP-{uuid.uuid4().hex[:6].upper()}"

class Ticket(Base):
    __tablename__ = "tickets"

    id = Column(String(36), primary_key=True, default=generate_uuid, index=True)
    tracking_code = Column(String(20), unique=True, index=True, default=generate_tracking_code)
    category = Column(String(100), nullable=False, index=True)
    department = Column(String(100), nullable=False, index=True)
    
    # Mathematical Severity & ML Breakdown
    severity_score = Column(Float, nullable=False, default=0.5)
    confidence_score = Column(Float, nullable=False, default=0.0)
    area_ratio = Column(Float, nullable=False, default=0.0)
    hazard_weight = Column(Float, nullable=False, default=0.0)
    weather_risk = Column(Float, nullable=False, default=0.0)
    
    # Geospatial Coordinates & Live Addresses
    latitude = Column(Float, nullable=False, index=True)
    longitude = Column(Float, nullable=False, index=True)
    address = Column(Text, nullable=True)
    ward_name = Column(String(100), nullable=True)
    
    # Triage State & Crowdsourced Escalation
    status = Column(String(30), default="OPEN", index=True)  # OPEN, IN_PROGRESS, VERIFIED_RESOLVED, REJECTED
    upvote_count = Column(Integer, default=1)
    sla_hours = Column(Float, nullable=False, default=48.0)
    sla_deadline = Column(DateTime, nullable=False)
    
    # Media & Visual Forensics
    original_image_path = Column(String(255), nullable=False)
    annotated_image_path = Column(String(255), nullable=True)
    verification_image_path = Column(String(255), nullable=True)
    # Video evidence support
    video_path = Column(String(255), nullable=True)            # Original uploaded video
    verification_video_path = Column(String(255), nullable=True)  # Proof-of-fix video
    media_type = Column(String(10), default="image")           # "image" or "video"

    trust_score = Column(Float, default=1.0)
    perceptual_hash = Column(String(64), index=True, nullable=True)
    
    # Rich Context Metadata (Stored as JSON Text)
    weather_details = Column(Text, nullable=True)
    detection_details = Column(Text, nullable=True)
    video_analysis_details = Column(Text, nullable=True)       # Video frame analysis results
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    resolved_at = Column(DateTime, nullable=True)

    # Relationships
    reports = relationship("CitizenReport", back_populates="ticket", cascade="all, delete-orphan", order_by="desc(CitizenReport.created_at)")
    audit_logs = relationship("AuditLog", back_populates="ticket", cascade="all, delete-orphan", order_by="desc(AuditLog.timestamp)")


class CitizenReport(Base):
    __tablename__ = "citizen_reports"

    id = Column(String(36), primary_key=True, default=generate_uuid, index=True)
    ticket_id = Column(String(36), ForeignKey("tickets.id", ondelete="CASCADE"), nullable=False, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    user_description = Column(Text, nullable=True)
    transcribed_text = Column(Text, nullable=True)
    urgency_keywords = Column(Text, nullable=True)  # JSON list
    image_path = Column(String(255), nullable=False)
    audio_path = Column(String(255), nullable=True)
    video_path = Column(String(255), nullable=True)            # Video evidence from citizen
    perceptual_hash = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    ticket = relationship("Ticket", back_populates="reports")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticket_id = Column(String(36), ForeignKey("tickets.id", ondelete="CASCADE"), nullable=False, index=True)
    action = Column(String(50), nullable=False)  # TICKET_CREATED, CROWD_ESCALATION, PROOF_OF_FIX_ACCEPTED, etc.
    details = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    ticket = relationship("Ticket", back_populates="audit_logs")
