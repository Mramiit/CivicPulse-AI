from datetime import datetime, timezone, timedelta
from typing import Tuple, Dict, Any

DEPARTMENT_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "Public Works Department (PWD)": {
        "keywords": ["pothole", "road", "asphalt", "crater", "crack", "footpath", "sidewalk", "divider", "pavement"],
        "primary_contact": "pwd-grievance@municipal.gov"
    },
    "Jal Board & Drainage Authority": {
        "keywords": ["waterlogging", "manhole", "drain", "drainage", "sewer", "sewage", "flood", "leakage", "pipeline", "gutter"],
        "primary_contact": "jalboard-urgent@municipal.gov"
    },
    "Municipal Solid Waste Management (SWM)": {
        "keywords": ["garbage", "trash", "waste", "dump", "debris", "litter", "rubbish", "stench", "overflow"],
        "primary_contact": "swm-triage@municipal.gov"
    },
    "Electricity & Power Distribution (DISCOM)": {
        "keywords": ["wire", "electric", "pole", "spark", "streetlight", "transformer", "shock", "cable"],
        "primary_contact": "discom-emergency@municipal.gov"
    }
}

def route_department(category: str, text_context: str = "") -> str:
    """
    Routes the civic grievance to the correct municipal authority using category
    and NLP keyword semantic matching.
    """
    combined_text = f"{category} {text_context}".lower()

    # Rule-based priority matching for critical hazards
    if any(k in combined_text for k in ["manhole", "waterlog", "sewer", "drain", "water"]):
        return "Jal Board & Drainage Authority"
    if any(k in combined_text for k in ["wire", "electric", "streetlight", "spark", "shock", "pole"]):
        return "Electricity & Power Distribution (DISCOM)"
    if any(k in combined_text for k in ["garbage", "waste", "trash", "debris", "dump"]):
        return "Municipal Solid Waste Management (SWM)"
    if any(k in combined_text for k in ["pothole", "road", "asphalt", "crater", "pavement"]):
        return "Public Works Department (PWD)"

    # Keyword scoring fallback
    best_dept = "Public Works Department (PWD)"
    highest_matches = 0
    
    for dept_name, info in DEPARTMENT_TAXONOMY.items():
        matches = sum(1 for kw in info["keywords"] if kw in combined_text)
        if matches > highest_matches:
            highest_matches = matches
            best_dept = dept_name

    return best_dept

def calculate_initial_sla(severity_score: float) -> Tuple[float, datetime]:
    """
    Calculates initial SLA resolution window (in hours) and exact deadline timestamp
    based on the multimodal severity score.
    
    - Critical (S >= 0.75): 18.0 hours
    - High (0.55 <= S < 0.75): 36.0 hours
    - Moderate (0.35 <= S < 0.55): 72.0 hours
    - Low (S < 0.35): 120.0 hours (5 days)
    """
    now = datetime.now(timezone.utc)
    
    if severity_score >= 0.75:
        sla_hours = 18.0
    elif severity_score >= 0.55:
        sla_hours = 36.0
    elif severity_score >= 0.35:
        sla_hours = 72.0
    else:
        sla_hours = 120.0

    deadline = now + timedelta(hours=sla_hours)
    return sla_hours, deadline
