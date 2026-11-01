from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

class DetectionBox(BaseModel):
    label: str
    confidence: float
    bbox: List[int] = Field(description="[x1, y1, x2, y2]")
    area_ratio: float

class SeverityBreakdown(BaseModel):
    confidence_term: float
    area_term: float
    hazard_term: float
    weather_term: float
    final_severity: float
    risk_level: str  # Critical, High, Moderate, Low
    explanation: str

class AuditLogOut(BaseModel):
    id: int
    action: str
    details: Optional[str] = None
    timestamp: datetime
    model_config = ConfigDict(from_attributes=True)

class CitizenReportOut(BaseModel):
    id: str
    ticket_id: str
    latitude: float
    longitude: float
    user_description: Optional[str] = None
    transcribed_text: Optional[str] = None
    urgency_keywords: Optional[str] = None
    image_path: str
    audio_path: Optional[str] = None
    video_path: Optional[str] = None
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

class TicketOut(BaseModel):
    id: str
    tracking_code: str
    category: str
    department: str
    severity_score: float
    confidence_score: float
    area_ratio: float
    hazard_weight: float
    weather_risk: float
    latitude: float
    longitude: float
    address: Optional[str] = None
    ward_name: Optional[str] = None
    status: str
    upvote_count: int
    sla_hours: float
    sla_deadline: datetime
    sla_remaining_hours: float
    sla_status: str  # ON_TRACK, AT_RISK, BREACHED
    media_type: str = "image"
    original_image_url: str
    annotated_image_url: Optional[str] = None
    verification_image_url: Optional[str] = None
    video_url: Optional[str] = None
    verification_video_url: Optional[str] = None
    trust_score: float
    perceptual_hash: Optional[str] = None
    weather_details: Optional[Dict[str, Any]] = None
    detection_details: Optional[List[Dict[str, Any]]] = None
    video_analysis: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None
    reports_count: int
    reports: List[CitizenReportOut] = []
    audit_logs: List[AuditLogOut] = []
    model_config = ConfigDict(from_attributes=True)

class TicketGeoJSONFeature(BaseModel):
    type: str = "Feature"
    geometry: Dict[str, Any]
    properties: Dict[str, Any]

class TicketGeoJSONCollection(BaseModel):
    type: str = "FeatureCollection"
    features: List[TicketGeoJSONFeature]

class VerifyFixResponse(BaseModel):
    ticket_id: str
    tracking_code: str
    status: str
    verified: bool
    initial_severity: float
    after_severity: float
    reduction_percentage: float
    detected_defects_after: int
    annotated_after_image_url: Optional[str] = None
    verification_video_url: Optional[str] = None
    message: str

class AnalyticsResponse(BaseModel):
    total_tickets: int
    active_tickets: int
    resolved_tickets: int
    avg_severity: float
    sla_breach_risk: Dict[str, int]  # {"critical_under_24h": X, "at_risk_24_48h": Y, "healthy": Z}
    department_load: Dict[str, int]
    category_distribution: Dict[str, int]
    recent_activity: List[Dict[str, Any]]
