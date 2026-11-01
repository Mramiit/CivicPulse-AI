from typing import Dict, Any, Tuple
import numpy as np

# Hazard Criticality Weight Taxonomy (W_hazard in [0.0, 1.0])
HAZARD_CRITICALITY: Dict[str, float] = {
    "Open Manhole / Drain": 1.00,             # Immediate catastrophic life-safety hazard
    "Fallen Tree / Road Hazard": 0.90,        # Complete traffic blockage & potential live wire hazard
    "Waterlogging & Drainage": 0.85,          # Vector disease, road foundation erosion & vehicle stalling
    "Streetlight / Electrical Hazard": 0.80,  # Electrocution risk and nighttime crime/accident exposure
    "Pothole / Road Damage": 0.75,            # Two-wheeler skid risk and structural road degradation
    "Solid Waste & Garbage": 0.50,            # Public sanitation and pest infestation hazard
    "General Civic Defect": 0.50              # Default baseline
}

def get_hazard_weight(category: str) -> float:
    for cat_name, weight in HAZARD_CRITICALITY.items():
        if cat_name.lower() in category.lower() or category.lower() in cat_name.lower():
            return weight
    return HAZARD_CRITICALITY["General Civic Defect"]

def calculate_multimodal_severity(
    c_conf: float,
    r_area: float,
    category: str,
    e_weather: float
) -> Tuple[float, Dict[str, Any]]:
    """
    Computes dynamic multimodal infrastructure severity:
    S = clip(0.30 * C_conf + 0.35 * R_area + 0.20 * W_hazard + 0.15 * E_weather, 0, 1)

    Includes Visual Evidence Verification Guard:
    If visual confidence C_conf == 0 and area ratio R_area == 0 (clean image / normal scene),
    unverified text or dropdown claims cannot escalate to High or Critical.
    Severity is anchored to LOW priority pending municipal desk review.
    """
    w_hazard = get_hazard_weight(category)
    
    # Ensure inputs are sanitized in [0, 1]
    c_conf = float(np.clip(c_conf, 0.0, 1.0))
    r_area = float(np.clip(r_area, 0.0, 1.0))
    w_hazard = float(np.clip(w_hazard, 0.0, 1.0))
    e_weather = float(np.clip(e_weather, 0.0, 1.0))

    # ── Visual Evidence Verification Guard ────────────────────────────────────
    if c_conf == 0.0 and r_area == 0.0:
        raw_score = 0.12 * w_hazard + 0.08 * e_weather
        final_score = float(np.clip(raw_score, 0.05, 0.25))
        final_score = round(final_score, 3)

        breakdown = {
            "confidence_raw": 0.0,
            "confidence_term": 0.0,
            "area_ratio_raw": 0.0,
            "area_term": 0.0,
            "hazard_weight_raw": round(w_hazard, 3),
            "hazard_term": round(0.12 * w_hazard, 3),
            "weather_risk_raw": round(e_weather, 3),
            "weather_term": round(0.08 * e_weather, 3),
            "final_severity": final_score,
            "risk_level": "LOW",
            "explanation": "No defect detected in visual evidence (clean/stable scene). Triage prioritized at Low pending municipal field verification.",
            "formula": "S_unverified = min(0.25, 0.12*W_hazard + 0.08*E_weather)"
        }
        return final_score, breakdown

    # ── Standard Multimodal Severity Calculation ──────────────────────────────
    term_conf = 0.30 * c_conf
    term_area = 0.35 * r_area
    term_hazard = 0.20 * w_hazard
    term_weather = 0.15 * e_weather

    raw_score = term_conf + term_area + term_hazard + term_weather
    final_score = float(np.clip(raw_score, 0.0, 1.0))
    final_score = round(final_score, 3)

    if final_score >= 0.70:
        risk_level = "CRITICAL"
        explanation = "Severe infrastructure emergency requiring immediate dispatch within 12-24 hours."
    elif final_score >= 0.50:
        risk_level = "HIGH"
        explanation = "High priority civic defect with substantial safety or traffic disruption potential."
    elif final_score >= 0.35:
        risk_level = "MODERATE"
        explanation = "Moderate maintenance item assigned to standard triage queue."
    else:
        risk_level = "LOW"
        explanation = "Minor cosmetic or non-critical surface maintenance defect."

    breakdown = {
        "confidence_raw": round(c_conf, 3),
        "confidence_term": round(term_conf, 3),
        "area_ratio_raw": round(r_area, 3),
        "area_term": round(term_area, 3),
        "hazard_weight_raw": round(w_hazard, 3),
        "hazard_term": round(term_hazard, 3),
        "weather_risk_raw": round(e_weather, 3),
        "weather_term": round(term_weather, 3),
        "final_severity": final_score,
        "risk_level": risk_level,
        "explanation": explanation,
        "formula": "S = clip(0.30*C_conf + 0.35*R_area + 0.20*W_hazard + 0.15*E_weather, 0, 1)"
    }

    return final_score, breakdown
