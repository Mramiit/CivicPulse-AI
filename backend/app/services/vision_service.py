import os
import io
import base64
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Defensive imports for cv2 and ultralytics
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

# Global YOLO model cache
_YOLO_MODEL = None

def get_yolo_model():
    global _YOLO_MODEL
    if _YOLO_MODEL is None and HAS_YOLO:
        try:
            # Check if custom weights exist, else load standard yolov8n
            weights_file = settings.YOLO_MODEL_PATH
            if not os.path.exists(weights_file):
                weights_file = "yolov8n.pt"  # Will auto-download small pretrained model if needed
            _YOLO_MODEL = YOLO(weights_file)
        except Exception as e:
            print(f"[VisionService] Warning: Could not initialize YOLO model: {e}")
            _YOLO_MODEL = None
    return _YOLO_MODEL


def detect_structural_contours_opencv(image_path: str) -> List[Dict[str, Any]]:
    """
    OpenCV contour and edge-density analyzer for surface defects (potholes, waterlogging,
    cracks, manholes) when YOLO object detection doesn't recognize non-standard civic defects.
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
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge detection & morphological dilation
        edges = cv2.Canny(blurred, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter out tiny noise and massive whole-image borders
            if area > (0.015 * total_area) and area < (0.85 * total_area):
                x, y, bw, bh = cv2.boundingRect(cnt)
                area_ratio = float(area / total_area)
                confidence = float(min(0.85, 0.40 + (area_ratio * 1.2)))
                defects.append({
                    "label": "Road Surface Anomaly / Defect",
                    "confidence": round(confidence, 2),
                    "bbox": [int(x), int(y), int(x + bw), int(y + bh)],
                    "area_ratio": round(area_ratio, 3)
                })
    except Exception as e:
        print(f"[VisionService] Contour detection error: {e}")

    return defects


def detect_structural_contours_pillow(image_path: str) -> List[Dict[str, Any]]:
    """
    Pillow/NumPy gradient fallback when OpenCV is not installed.
    Computes intensity gradients to identify dark/high-contrast defect regions.
    """
    defects = []
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            total_area = float(w * h)
            img_gray = img.convert("L").resize((128, 128))
            arr = np.array(img_gray, dtype=np.float32)

            # Gradient magnitude
            gy, gx = np.gradient(arr)
            grad = np.sqrt(gx**2 + gy**2)
            threshold = np.percentile(grad, 88)
            mask = grad > threshold

            if np.mean(mask) > 0.05:
                # Estimate central defect bounding box
                y_indices, x_indices = np.where(mask)
                if len(x_indices) > 0 and len(y_indices) > 0:
                    scale_x = w / 128.0
                    scale_y = h / 128.0
                    x1 = int(np.percentile(x_indices, 10) * scale_x)
                    x2 = int(np.percentile(x_indices, 90) * scale_x)
                    y1 = int(np.percentile(y_indices, 10) * scale_y)
                    y2 = int(np.percentile(y_indices, 90) * scale_y)

                    box_area = (x2 - x1) * (y2 - y1)
                    area_ratio = min(0.70, max(0.04, float(box_area / total_area)))
                    confidence = 0.65
                    defects.append({
                        "label": "Structural Road Defect",
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2],
                        "area_ratio": round(area_ratio, 3)
                    })
    except Exception as e:
        print(f"[VisionService] Pillow fallback error: {e}")
    return defects


def annotate_image(image_path: str, detections: List[Dict[str, Any]], output_path: str) -> str:
    """
    Renders bounding boxes and label tags onto the image and saves it to output_path.
    Returns base64 encoded data URI string.
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            draw = ImageDraw.Draw(img)
            
            for det in detections:
                bbox = det.get("bbox", [0, 0, 0, 0])
                label = det.get("label", "Defect")
                conf = det.get("confidence", 0.0)
                
                x1, y1, x2, y2 = bbox
                # Draw thick bounding box
                draw.rectangle([x1, y1, x2, y2], outline="#EF4444", width=4)
                
                # Tag text
                tag_text = f"{label} ({int(conf * 100)}%)"
                # Draw background rectangle for label
                draw.rectangle([x1, max(0, y1 - 22), x1 + len(tag_text) * 8 + 12, y1], fill="#EF4444")
                draw.text((x1 + 6, max(0, y1 - 18)), tag_text, fill="#FFFFFF")

            img.save(output_path, "JPEG", quality=88)

            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{img_b64}"
    except Exception as e:
        print(f"[VisionService] Annotation error: {e}")
        return ""


def run_vision_pipeline(image_path: str, annotated_output_path: Optional[str] = None) -> Tuple[List[Dict[str, Any]], float, float, str]:
    """
    Executes the multimodal vision pipeline:
    1. YOLOv8 object detection
    2. Structural contour & edge-density fallback
    3. Area-ratio & aggregate confidence calculation
    4. Annotated image generation

    Returns:
        Tuple of (detections_list, c_conf, r_area, base64_annotated_preview)
    """
    detections: List[Dict[str, Any]] = []
    
    # 1. Attempt YOLO detection
    model = get_yolo_model()
    if model is not None:
        try:
            results = model(image_path, verbose=False)
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    img_h, img_w = r.orig_shape
                    img_area = float(img_h * img_w)
                    for box in boxes:
                        coords = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        class_name = model.names.get(cls_id, "Defect")
                        
                        x1, y1, x2, y2 = [int(v) for v in coords]
                        box_area = float((x2 - x1) * (y2 - y1))
                        area_ratio = box_area / img_area
                        
                        detections.append({
                            "label": class_name.title(),
                            "confidence": round(conf, 2),
                            "bbox": [x1, y1, x2, y2],
                            "area_ratio": round(area_ratio, 3)
                        })
        except Exception as e:
            print(f"[VisionService] YOLO inference notice: {e}")

    # 2. Structural contour fallback if YOLO returned no detections
    if not detections:
        if HAS_CV2:
            detections = detect_structural_contours_opencv(image_path)
        else:
            detections = detect_structural_contours_pillow(image_path)

    # 3. Compute aggregate C_conf and R_area
    if detections:
        c_conf = float(max(d["confidence"] for d in detections))
        r_area = float(min(1.0, sum(d["area_ratio"] for d in detections)))
    else:
        # Default baseline if visual defect is subtle
        c_conf = 0.40
        r_area = 0.10

    # 4. Generate annotated preview
    base64_preview = ""
    if annotated_output_path:
        base64_preview = annotate_image(image_path, detections, annotated_output_path)

    return detections, round(c_conf, 2), round(r_area, 2), base64_preview
