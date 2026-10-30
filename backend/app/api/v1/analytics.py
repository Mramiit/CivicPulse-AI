from datetime import datetime, timezone
from typing import Dict, Any
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.app.api.deps import get_db
from backend.app.models.ticket import Ticket, AuditLog

router = APIRouter()

@router.get("/analytics")
def get_analytics(db: Session = Depends(get_db)):
    """
    Computes real-time municipal triage analytics:
    - Total, active, and resolved counts
    - Mean severity index
    - SLA breach risk counts
    - Department workload distribution
    - Defect category distribution
    - Live audit trail stream
    """
    total_tickets = db.query(Ticket).count()
    active_tickets = db.query(Ticket).filter(Ticket.status.in_(["OPEN", "IN_PROGRESS"])).count()
    resolved_tickets = db.query(Ticket).filter(Ticket.status == "VERIFIED_RESOLVED").count()

    # Average severity of active tickets
    avg_severity = db.query(func.avg(Ticket.severity_score)).filter(Ticket.status.in_(["OPEN", "IN_PROGRESS"])).scalar() or 0.0
    avg_severity = round(float(avg_severity), 3)

    # SLA breach analysis
    now = datetime.now(timezone.utc)
    active_records = db.query(Ticket).filter(Ticket.status.in_(["OPEN", "IN_PROGRESS"])).all()

    critical_under_24 = 0
    at_risk_24_48 = 0
    healthy = 0

    for t in active_records:
        deadline = t.sla_deadline
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        rem_hours = (deadline - now).total_seconds() / 3600.0
        if rem_hours <= 24.0:
            critical_under_24 += 1
        elif rem_hours <= 48.0:
            at_risk_24_48 += 1
        else:
            healthy += 1

    # Department distribution
    dept_counts = (
        db.query(Ticket.department, func.count(Ticket.id))
        .group_by(Ticket.department)
        .all()
    )
    department_load = {dept: count for dept, count in dept_counts}

    # Category distribution
    cat_counts = (
        db.query(Ticket.category, func.count(Ticket.id))
        .group_by(Ticket.category)
        .all()
    )
    category_distribution = {cat: count for cat, count in cat_counts}

    # Recent Audit Log Activity
    recent_logs = (
        db.query(AuditLog, Ticket.tracking_code)
        .join(Ticket, AuditLog.ticket_id == Ticket.id)
        .order_by(AuditLog.timestamp.desc())
        .limit(8)
        .all()
    )

    activity = [
        {
            "id": log.id,
            "tracking_code": tracking_code,
            "action": log.action,
            "details": log.details,
            "timestamp": log.timestamp.isoformat()
        }
        for log, tracking_code in recent_logs
    ]

    return {
        "total_tickets": total_tickets,
        "active_tickets": active_tickets,
        "resolved_tickets": resolved_tickets,
        "avg_severity": avg_severity,
        "sla_breach_risk": {
            "critical_under_24h": critical_under_24,
            "at_risk_24_48h": at_risk_24_48,
            "healthy": healthy
        },
        "department_load": department_load,
        "category_distribution": category_distribution,
        "recent_activity": activity
    }
