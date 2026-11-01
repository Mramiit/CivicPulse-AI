"""
CivicPulse AI — 3-Layer Civic Defect Vision Pipeline
======================================================
Architecture:
  Layer 1: YOLOv8 (civic fine-tuned > COCO pretrained)
           Detects bounding boxes: Potholes, Cracks, Manholes, Garbage.
  Layer 2: MobileNetV2 CNN Classifier
           Confirms/relabels detections; identifies "Normal / No Defect" scenes.
  Layer 3: OpenCV Structural Analysis (Canny + Adaptive Contours)
           Detects circular potholes vs linear cracks on asphalt road surfaces.

When an image depicts a normal building, clean scene, or stable infrastructure,
the pipeline accurately reports 0 defects (confidence=0.0, area=0.0) and annotates
with a clean verification badge, avoiding false emergency escalations.
"""

import os
import io
import base64
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

from backend.app.config import settings

# ─── Model Cache ──────────────────────────────────────────────────────────────
_YOLO_MODEL: Optional[object] = None

CIVIC_YOLO_CLASSES = {
    "pothole", "manhole", "drain", "garbage", "trash", "crack", "water",
    "streetlight", "fallen tree", "debris"
}

# Color per detection source (used in annotation)
_SOURCE_COLORS = {
    "yolo_civic":           "#10B981",  # Green  — civic fine-tuned (best)
    "yolo_coco+classifier": "#3B82F6",  # Blue   — COCO + classifier refinement
    "classifier":           "#8B5CF6",  # Purple — pure CNN classifier
    "opencv":               "#F59E0B",  # Amber  — OpenCV heuristic
    "pillow_gradient":      "#6B7280",  # Grey   — gradient fallback
}
_DEFAULT_COLOR = "#EF4444"


# =============================================================================
# LAYER 1 — YOLOv8 (civic fine-tuned > COCO filtered)
# =============================================================================

def get_yolo_model():
    """
    Loads YOLO with priority:
      1. weights/civicpulse_yolo.pt  — fine-tuned on RDD2022 (best)
      2. settings.YOLO_MODEL_PATH    — custom path from config
      3. yolov8n.pt                  — COCO pretrained (limited civic classes)
    """
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL

    if not HAS_YOLO:
        return None

    candidates = [
        os.path.join(settings.WEIGHTS_DIR, "civicpulse_yolo.pt"),
        settings.YOLO_MODEL_PATH,
        "yolov8n.pt",
    ]

    for weights_file in candidates:
        if not weights_file:
            continue
        if os.path.exists(weights_file) or weights_file == "yolov8n.pt":
            try:
                _YOLO_MODEL = YOLO(weights_file)
                tag = "civic fine-tuned" if "civicpulse" in weights_file else "COCO pretrained"
                print(f"[VisionService] YOLO loaded: {os.path.basename(weights_file)} ({tag})")
                return _YOLO_MODEL
            except Exception as e:
                print(f"[VisionService] Could not load {weights_file}: {e}")

    return None


def _is_civic_class(class_name: str) -> bool:
    return any(c in class_name.lower() for c in CIVIC_YOLO_CLASSES)


def run_yolo_detection(image_path: str) -> List[Dict[str, Any]]:
    """Runs YOLOv8. Keeps ALL classes for civic model; filters noise for COCO model."""
    detections = []
    model = get_yolo_model()
    if model is None:
        return detections

    is_civic_model = any(
        _is_civic_class(cls) for cls in (model.names or {}).values()
    )

    try:
        results = model(image_path, verbose=False, conf=0.25)
        for r in results:
            if r.boxes is None:
                continue
            img_h, img_w = r.orig_shape
            img_area = float(img_h * img_w)

            for box in r.boxes:
                conf       = float(box.conf[0])
                cls_id     = int(box.cls[0])
                class_name = model.names.get(cls_id, "Defect")

                # For COCO model: skip non-civic classes and low confidence
                if not is_civic_model:
                    if conf < 0.40 or not _is_civic_class(class_name):
                        continue

                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                box_area   = float((x2 - x1) * (y2 - y1))
                area_ratio = box_area / img_area

                detections.append({
                    "label":      class_name.title(),
                    "confidence": round(conf, 2),
                    "bbox":       [x1, y1, x2, y2],
                    "area_ratio": round(area_ratio, 3),
                    "source":     "yolo_civic" if is_civic_model else "yolo_coco",
                })

    except Exception as e:
        print(f"[VisionService] YOLO inference error: {e}")

    return detections


# =============================================================================
# LAYER 2 — MobileNetV2 CNN Classifier
# =============================================================================

def run_classifier_detection(
    image_path: str,
    yolo_detections: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Uses classifier_service.classify_civic_defect() to:
      - Confirm/relabel YOLO detections
      - Identify clean/normal scenes without synthesizing fake boxes
      - Only synthesize a localized detection if defect confidence is genuinely high
    """
    try:
        from backend.app.services.classifier_service import classify_civic_defect

        clf_weights = os.path.join(settings.WEIGHTS_DIR, "civicpulse_classifier.pt")
        predicted_class, confidence, all_scores = classify_civic_defect(
            image_path,
            weights_path=clf_weights if os.path.exists(clf_weights) else None
        )

        # If classifier identifies the scene as Normal / No Defect, do NOT add defect boxes
        if "normal" in predicted_class.lower() or predicted_class == "Normal / No Defect":
            # If YOLO also had nothing, this is a clean scene
            if not yolo_detections:
                return []
            return yolo_detections

        if yolo_detections:
            from_civic = any(d.get("source") == "yolo_civic" for d in yolo_detections)
            if from_civic:
                for d in yolo_detections:
                    d["classifier_class"] = predicted_class
                    d["classifier_conf"]  = confidence
                return yolo_detections
            else:
                for d in yolo_detections:
                    d["label"]      = predicted_class
                    d["confidence"] = round((d["confidence"] + confidence) / 2.0, 2)
                    d["source"]     = "yolo_coco+classifier"
                    d["classifier_class"] = predicted_class
                return yolo_detections

        # YOLO found nothing: ONLY synthesize a localized box if defect confidence is HIGH (>= 0.70)
        if confidence >= 0.70:
            try:
                with Image.open(image_path) as img:
                    w, h = img.size
            except Exception:
                w, h = 640, 480

            # Localized ground-level bounding box (bottom 50% of image), NOT full-screen
            bbox = [int(w * 0.20), int(h * 0.45), int(w * 0.80), int(h * 0.90)]
            area_ratio = round((0.60 * 0.45), 3)  # Realistic localized defect area (~0.27)

            top_scores = dict(sorted(all_scores.items(), key=lambda x: -x[1])[:4])
            return [{
                "label":      predicted_class,
                "confidence": round(confidence, 2),
                "bbox":       bbox,
                "area_ratio": area_ratio,
                "source":     "classifier",
                "all_scores": top_scores,
            }]

        return yolo_detections

    except Exception as e:
        print(f"[VisionService] Classifier layer error: {e}")
        return yolo_detections


# =============================================================================
# LAYER 3 — OpenCV / Pillow Structural Analysis (final fallback)
# =============================================================================

def detect_structural_contours_opencv(image_path: str) -> List[Dict[str, Any]]:
    """
    OpenCV analysis: Canny edges + adaptive dark-region detection.
    Restricted to lower road region; skips architectural buildings to prevent false alarms.
    """
    defects = []
    if not HAS_CV2 or not os.path.exists(image_path):
        return defects

    try:
        img = cv2.imread(image_path)
        if img is None:
            return defects

        h, w = img.shape[:2]
        total_area = float(h * w)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Architectural Building Guard: check edge alignment
        # If dominant edges are strictly horizontal and vertical (windows, roofs, walls),
        # do NOT flag building features as road surface defects.
        edges_full = cv2.Canny(gray, 50, 150)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        sig = mag > 25.0
        if np.sum(sig) > 50:
            angles = np.abs(np.arctan2(np.abs(gy[sig]), np.abs(gx[sig]) + 1e-6))
            axis_aligned = np.mean((angles < 0.26) | (angles > 1.31))
            if axis_aligned > 0.58:
                # Building facade / architectural grid detected — skip contour defect false positives
                return []

        # Only analyze lower 65% of image (road level)
        road_start_y = int(h * 0.35)
        road_gray = gray[road_start_y:, :]
        blur = cv2.GaussianBlur(road_gray, (5, 5), 0)

        edges  = cv2.Canny(blur, 40, 120)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Dark blob detection (potholes appear dark on lighter asphalt)
        _, dark_mask = cv2.threshold(blur, 75, 255, cv2.THRESH_BINARY_INV)
        dark_clean   = cv2.morphologyEx(
            dark_mask, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )

        combined = cv2.bitwise_or(closed, dark_clean)
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Must be a realistic localized defect (between 1.5% and 40% of road surface)
            if area < (0.015 * total_area) or area > (0.40 * total_area):
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            real_y = y + road_start_y
            area_ratio   = float(area / total_area)
            perimeter    = cv2.arcLength(cnt, True)
            circularity  = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
            is_pothole   = circularity > 0.30

            confidence = float(min(0.78, 0.35 + area_ratio * 1.1 + circularity * 0.15))
            label      = "Pothole / Road Damage" if is_pothole else "Road Surface Crack / Anomaly"

            defects.append({
                "label":      label,
                "confidence": round(confidence, 2),
                "bbox":       [int(x), int(real_y), int(x + bw), int(real_y + bh)],
                "area_ratio": round(area_ratio, 3),
                "source":     "opencv",
            })

    except Exception as e:
        print(f"[VisionService] OpenCV analysis error: {e}")

    return defects


def detect_structural_contours_pillow(image_path: str) -> List[Dict[str, Any]]:
    """Pillow/NumPy gradient fallback when OpenCV is not installed."""
    defects = []
    try:
        with Image.open(image_path) as img:
            w, h       = img.size
            total_area = float(w * h)
            gray_sm    = img.convert("L").resize((128, 128))
            arr        = np.array(gray_sm, dtype=np.float32)

            gy, gx = np.gradient(arr)
            grad      = np.sqrt(gx**2 + gy**2)
            threshold = np.percentile(grad, 90)
            mask      = grad > threshold

            if np.mean(mask) > 0.05:
                y_idx, x_idx = np.where(mask)
                if len(x_idx) > 0:
                    sx, sy = w / 128.0, h / 128.0
                    x1 = int(np.percentile(x_idx, 15) * sx)
                    x2 = int(np.percentile(x_idx, 85) * sx)
                    y1 = int(np.percentile(y_idx, 40) * sy)  # Lower region only
                    y2 = int(np.percentile(y_idx, 90) * sy)

                    box_area   = (x2 - x1) * (y2 - y1)
                    area_ratio = min(0.35, max(0.04, float(box_area / total_area)))
                    defects.append({
                        "label":      "Structural Road Defect",
                        "confidence": 0.45,
                        "bbox":       [x1, y1, x2, y2],
                        "area_ratio": round(area_ratio, 3),
                        "source":     "pillow_gradient",
                    })

    except Exception as e:
        print(f"[VisionService] Pillow contour error: {e}")

    return defects


# =============================================================================
# Image Annotation
# =============================================================================

def annotate_image(
    image_path: str,
    detections: List[Dict[str, Any]],
    output_path: str
) -> str:
    """
    Renders color-coded bounding boxes onto the image.
    If no defect is detected, displays a clean green 'Verified Stable Scene' badge.
    """
    try:
        with Image.open(image_path) as img:
            img  = img.convert("RGB")
            draw = ImageDraw.Draw(img)
            w, h = img.size

            if not detections:
                # Clean Scene Badge (No Defect Found)
                banner_w = min(w - 20, 360)
                draw.rectangle([10, 10, 10 + banner_w, 36], fill="#059669")
                draw.text((18, 15), "✓ AI Vision Audit: Stable / No Defect Detected", fill="#FFFFFF")
            else:
                for det in detections:
                    bbox   = det.get("bbox", [0, 0, 0, 0])
                    label  = det.get("label", "Defect")
                    conf   = det.get("confidence", 0.0)
                    source = det.get("source", "")
                    color  = _SOURCE_COLORS.get(source, _DEFAULT_COLOR)

                    x1, y1, x2, y2 = bbox
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

                    tag  = f"{label} ({int(conf * 100)}%)"
                    tw   = len(tag) * 8 + 12
                    draw.rectangle([x1, max(0, y1 - 22), x1 + tw, y1], fill=color)
                    draw.text((x1 + 6, max(0, y1 - 18)), tag, fill="#FFFFFF")

            img.save(output_path, "JPEG", quality=88)
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

    except Exception as e:
        print(f"[VisionService] Annotation error: {e}")
        return ""


# =============================================================================
# Main Entry Point
# =============================================================================

def run_vision_pipeline(
    image_path: str,
    annotated_output_path: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], float, float, str]:
    """
    3-Layer Civic Defect Vision Pipeline
    =====================================
    L1 — YOLOv8       : civic fine-tuned weights (or COCO filtered)
    L2 — CNN Classifier: MobileNetV2 / intelligent scene analysis
    L3 — OpenCV/Pillow : structural edge + dark-blob analysis (road surface only)

    Returns: (detections, c_conf, r_area, base64_annotated_preview)
    """
    # Layer 1 — YOLO
    detections = run_yolo_detection(image_path)
    print(f"[VisionPipeline] L1 YOLO: {len(detections)} detection(s)")

    # Layer 2 — CNN Classifier
    detections = run_classifier_detection(image_path, detections)
    print(f"[VisionPipeline] L2 Classifier: {len(detections)} detection(s)")

    # Layer 3 — Structural analysis (only when L1+L2 both yield nothing)
    if not detections:
        if HAS_CV2:
            detections = detect_structural_contours_opencv(image_path)
        else:
            detections = detect_structural_contours_pillow(image_path)
        print(f"[VisionPipeline] L3 Structural: {len(detections)} detection(s)")

    # Aggregate scores (accurate 0.0 when no defect exists)
    if detections:
        c_conf = float(max(d["confidence"] for d in detections))
        r_area = float(min(1.0, sum(d["area_ratio"] for d in detections)))
    else:
        c_conf = 0.0
        r_area = 0.0

    # Annotate
    base64_preview = ""
    if annotated_output_path:
        base64_preview = annotate_image(image_path, detections, annotated_output_path)

    return detections, round(c_conf, 2), round(r_area, 2), base64_preview
