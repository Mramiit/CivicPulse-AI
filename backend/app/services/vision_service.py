import os
import io
import base64
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw

# ── Defensive imports ─────────────────────────────────────────────────────────
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

# ── Model cache ───────────────────────────────────────────────────────────────
_YOLO_MODEL = None

# Civic defect class keywords (used by the fine-tuned RDD2022 model)
CIVIC_YOLO_CLASSES = {
    "longitudinal crack", "transverse crack", "alligator crack", "pothole",
    "road damage", "crack", "manhole", "garbage", "waterlog",
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

    # Detect whether model is civic fine-tuned
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
      - Confirm/relabel YOLO detections (when YOLO is COCO-pretrained)
      - Generate a detection entry when YOLO found nothing

    Works zero-shot immediately; improves further after training.
    """
    try:
        from backend.app.services.classifier_service import classify_civic_defect

        clf_weights = os.path.join(settings.WEIGHTS_DIR, "civicpulse_classifier.pt")
        predicted_class, confidence, all_scores = classify_civic_defect(
            image_path,
            weights_path=clf_weights if os.path.exists(clf_weights) else None
        )

        # If classifier says "normal" with low confidence, don't add noise
        if "normal" in predicted_class.lower() and confidence < 0.55:
            return yolo_detections

        if yolo_detections:
            from_civic = any(d.get("source") == "yolo_civic" for d in yolo_detections)
            if from_civic:
                # Civic YOLO is authoritative; classifier just annotates
                for d in yolo_detections:
                    d["classifier_class"] = predicted_class
                    d["classifier_conf"]  = confidence
                return yolo_detections
            else:
                # COCO YOLO detections — relabel with classifier
                for d in yolo_detections:
                    d["label"]      = predicted_class
                    d["confidence"] = round((d["confidence"] + confidence) / 2.0, 2)
                    d["source"]     = "yolo_coco+classifier"
                    d["classifier_class"] = predicted_class
                return yolo_detections

        # YOLO found nothing — synthesise a full-image detection from classifier
        try:
            with Image.open(image_path) as img:
                w, h = img.size
        except Exception:
            w, h = 640, 480

        bbox       = [int(w * 0.05), int(h * 0.05), int(w * 0.95), int(h * 0.95)]
        area_ratio = round(min(0.90, 0.90 * 0.90), 3)

        top_scores = dict(sorted(all_scores.items(), key=lambda x: -x[1])[:4])
        return [{
            "label":      predicted_class,
            "confidence": round(confidence, 2),
            "bbox":       bbox,
            "area_ratio": area_ratio,
            "source":     "classifier",
            "all_scores": top_scores,
        }]

    except Exception as e:
        print(f"[VisionService] Classifier layer error: {e}")
        return yolo_detections


# =============================================================================
# LAYER 3 — OpenCV / Pillow Structural Analysis (final fallback)
# =============================================================================

def detect_structural_contours_opencv(image_path: str) -> List[Dict[str, Any]]:
    """
    Improved OpenCV analysis: Canny edges + adaptive dark-region detection.
    Circularity test distinguishes potholes (round) from cracks (linear).
    """
    defects = []
    if not HAS_CV2 or not os.path.exists(image_path):
        return defects

    try:
        img = cv2.imread(image_path)
        if img is None:
            return defects

        h, w       = img.shape[:2]
        total_area = float(h * w)
        gray       = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur       = cv2.GaussianBlur(gray, (5, 5), 0)

        # Technique A: Canny + morphological close
        edges  = cv2.Canny(blur, 40, 120)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Technique B: Dark blob detection (potholes appear dark on lighter asphalt)
        _, dark_mask = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY_INV)
        dark_clean   = cv2.morphologyEx(
            dark_mask, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )

        combined   = cv2.bitwise_or(closed, dark_clean)
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours   = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (0.01 * total_area) or area > (0.80 * total_area):
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            area_ratio   = float(area / total_area)
            perimeter    = cv2.arcLength(cnt, True)
            circularity  = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
            is_pothole   = circularity > 0.30

            confidence = float(min(0.82, 0.38 + area_ratio * 1.1 + circularity * 0.15))
            label      = "Pothole / Road Damage" if is_pothole else "Road Surface Crack / Anomaly"

            defects.append({
                "label":      label,
                "confidence": round(confidence, 2),
                "bbox":       [int(x), int(y), int(x + bw), int(y + bh)],
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
            threshold = np.percentile(grad, 87)
            mask      = grad > threshold

            if np.mean(mask) > 0.04:
                y_idx, x_idx = np.where(mask)
                if len(x_idx) > 0:
                    sx, sy = w / 128.0, h / 128.0
                    x1 = int(np.percentile(x_idx, 10) * sx)
                    x2 = int(np.percentile(x_idx, 90) * sx)
                    y1 = int(np.percentile(y_idx, 10) * sy)
                    y2 = int(np.percentile(y_idx, 90) * sy)

                    box_area   = (x2 - x1) * (y2 - y1)
                    area_ratio = min(0.70, max(0.04, float(box_area / total_area)))
                    defects.append({
                        "label":      "Structural Road Defect",
                        "confidence": 0.52,
                        "bbox":       [x1, y1, x2, y2],
                        "area_ratio": round(area_ratio, 3),
                        "source":     "pillow_gradient",
                    })
    except Exception as e:
        print(f"[VisionService] Pillow fallback error: {e}")
    return defects


# =============================================================================
# Annotation
# =============================================================================

def annotate_image(
    image_path: str,
    detections: List[Dict[str, Any]],
    output_path: str
) -> str:
    """
    Renders color-coded bounding boxes onto the image.
    Green = civic YOLO  |  Blue = COCO+classifier  |  Purple = classifier
    Amber = OpenCV      |  Grey  = gradient fallback
    """
    try:
        with Image.open(image_path) as img:
            img  = img.convert("RGB")
            draw = ImageDraw.Draw(img)

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
    L2 — CNN Classifier: MobileNetV2 (zero-shot now; improves after training)
    L3 — OpenCV/Pillow : structural edge + dark-blob analysis (always-on fallback)

    Returns: (detections, c_conf, r_area, base64_annotated_preview)
    """
    # Layer 1 — YOLO
    detections = run_yolo_detection(image_path)
    print(f"[VisionPipeline] L1 YOLO: {len(detections)} detection(s)")

    # Layer 2 — CNN Classifier (always runs to confirm/fill)
    detections = run_classifier_detection(image_path, detections)
    print(f"[VisionPipeline] L2 Classifier: {len(detections)} detection(s) "
          f"sources={[d.get('source', '?') for d in detections]}")

    # Layer 3 — Structural analysis (only when L1+L2 both yield nothing)
    if not detections:
        if HAS_CV2:
            detections = detect_structural_contours_opencv(image_path)
        else:
            detections = detect_structural_contours_pillow(image_path)
        print(f"[VisionPipeline] L3 Structural: {len(detections)} detection(s)")

    # Aggregate scores
    if detections:
        c_conf = float(max(d["confidence"] for d in detections))
        r_area = float(min(1.0, sum(d["area_ratio"] for d in detections)))
    else:
        c_conf = 0.35
        r_area = 0.08

    # Annotate
    base64_preview = ""
    if annotated_output_path:
        base64_preview = annotate_image(image_path, detections, annotated_output_path)

    return detections, round(c_conf, 2), round(r_area, 2), base64_preview
