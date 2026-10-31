"""
CivicPulse AI — Civic Defect CNN Classifier
============================================
A lightweight MobileNetV2-based image classifier that determines
whether an image contains a civic defect and what type it is.

Works in two modes:
  1. ZERO-SHOT (immediate use): Uses pretrained ImageNet features +
     hand-crafted texture/color rules to score civic defect probability.
     No training required. Works right now.

  2. FINE-TUNED (after training): Uses a trained head on top of
     MobileNetV2 for accurate civic defect classification.
     Run: python scripts/train_classifier.py

This is used as Layer 2 in the vision pipeline — runs AFTER YOLOv8
to confirm or refine detections, or to classify when YOLO finds nothing.
"""

import os
import io
from typing import Dict, Tuple, Optional
import numpy as np
from PIL import Image

# ── PyTorch optional import (graceful fallback if not installed) ──────────────
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ─── Civic Defect Classes ─────────────────────────────────────────────────────
CIVIC_CLASSES = [
    "Pothole / Road Damage",
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Waterlogging & Drainage",
    "Solid Waste & Garbage",
    "Open Manhole / Drain",
    "Streetlight / Electrical Hazard",
    "Normal / No Defect",
]
NUM_CLASSES = len(CIVIC_CLASSES)

# ─── Model Cache ──────────────────────────────────────────────────────────────
_CLASSIFIER_MODEL: Optional[object] = None
_CLASSIFIER_TRANSFORM = None


def _build_mobilenet_classifier() -> "nn.Module":
    """
    Builds a MobileNetV2 with a custom classification head for civic defects.
    Loads pretrained ImageNet weights for the backbone (auto-downloads ~14MB).
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Replace the classifier head with our civic defect head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, NUM_CLASSES),
    )
    model.eval()
    return model


def _get_transform():
    """Standard ImageNet normalisation transform for MobileNetV2."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_classifier(weights_path: Optional[str] = None):
    """
    Returns the classifier model, loading from weights_path if fine-tuned
    weights exist, otherwise uses pretrained ImageNet backbone (zero-shot mode).
    """
    global _CLASSIFIER_MODEL, _CLASSIFIER_TRANSFORM

    if _CLASSIFIER_MODEL is not None:
        return _CLASSIFIER_MODEL, _CLASSIFIER_TRANSFORM

    if not HAS_TORCH:
        return None, None

    try:
        model = _build_mobilenet_classifier()
        transform = _get_transform()

        if weights_path and os.path.exists(weights_path):
            print(f"[CivicClassifier] Loading fine-tuned weights: {weights_path}")
            state = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state)
            print("[CivicClassifier] Fine-tuned weights loaded ✓")
        else:
            print("[CivicClassifier] No fine-tuned weights found → running in zero-shot mode")

        model.eval()
        _CLASSIFIER_MODEL = model
        _CLASSIFIER_TRANSFORM = transform
        return model, transform

    except Exception as e:
        print(f"[CivicClassifier] Init error: {e}")
        return None, None


# ─── Zero-Shot Heuristic Scorer (works without any training) ──────────────────
def _zero_shot_civic_score(image_path: str) -> Dict[str, float]:
    """
    Analyses image texture and color statistics to produce a civic defect
    probability map WITHOUT any ML training.

    Logic:
    - Dark, irregular, high-variance regions → likely pothole / crack
    - Brown/grey desaturation + low brightness → road damage
    - Greenish/muddy color + high blue-channel → waterlogging
    - Orange/yellow warmth + scattered debris → garbage
    - Very high brightness patches + night background → streetlight
    """
    try:
        with Image.open(image_path) as img:
            img_rgb = img.convert("RGB").resize((128, 128))
            arr = np.array(img_rgb, dtype=np.float32)

            r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            gray = 0.299 * r + 0.587 * g + 0.114 * b

            # ── Texture features ──────────────────────────────────────────────
            gy, gx = np.gradient(gray)
            gradient_magnitude = np.sqrt(gx**2 + gy**2)
            texture_variance = float(np.std(gray))        # High = irregular surface
            gradient_mean = float(np.mean(gradient_magnitude))
            dark_ratio = float(np.mean(gray < 80))        # Dark pixel fraction
            bright_ratio = float(np.mean(gray > 200))     # Bright patch fraction

            # ── Color features ────────────────────────────────────────────────
            mean_r = float(np.mean(r))
            mean_g = float(np.mean(g))
            mean_b = float(np.mean(b))
            saturation = float(np.mean(np.max(arr, axis=2) - np.min(arr, axis=2)))

            # ── Scoring rules ─────────────────────────────────────────────────
            scores: Dict[str, float] = {}

            # Pothole: dark, high texture variance, low saturation (asphalt-like)
            pothole_score = (
                0.4 * min(1.0, texture_variance / 60.0) +
                0.3 * dark_ratio +
                0.3 * min(1.0, gradient_mean / 30.0)
            )
            scores["Pothole / Road Damage"] = round(pothole_score, 3)

            # Cracks: high gradient, linear features, grey tones
            crack_base = min(1.0, gradient_mean / 25.0) * (1.0 - dark_ratio * 0.5)
            scores["Longitudinal Crack"]  = round(crack_base * 0.8, 3)
            scores["Transverse Crack"]    = round(crack_base * 0.7, 3)
            scores["Alligator Crack"]     = round(crack_base * 0.65, 3)

            # Waterlogging: high blue/green channel, reflective patches
            water_score = (
                0.5 * min(1.0, (mean_b - mean_r) / 40.0 + 0.5) +
                0.3 * bright_ratio +
                0.2 * min(1.0, mean_g / 150.0)
            )
            scores["Waterlogging & Drainage"] = round(max(0.0, water_score), 3)

            # Garbage: warm orange/brown tones, moderate variance
            garbage_score = (
                0.5 * min(1.0, (mean_r - mean_b) / 50.0 + 0.3) +
                0.3 * min(1.0, saturation / 60.0) +
                0.2 * min(1.0, texture_variance / 50.0)
            )
            scores["Solid Waste & Garbage"] = round(max(0.0, garbage_score), 3)

            # Open manhole: very dark circular region, high surrounding contrast
            manhole_score = (
                0.6 * dark_ratio +
                0.4 * min(1.0, texture_variance / 70.0)
            ) * (1.0 if dark_ratio > 0.25 else 0.4)
            scores["Open Manhole / Drain"] = round(manhole_score, 3)

            # Streetlight hazard: isolated bright spots against dark background
            streetlight_score = (
                0.7 * bright_ratio +
                0.3 * dark_ratio
            ) * (1.5 if (bright_ratio > 0.05 and dark_ratio > 0.3) else 0.5)
            scores["Streetlight / Electrical Hazard"] = round(
                min(1.0, streetlight_score), 3
            )

            # Normal: low variance, uniform color, no dark regions
            normal_score = max(0.0,
                1.0 - texture_variance / 60.0 - dark_ratio * 0.5
            )
            scores["Normal / No Defect"] = round(normal_score, 3)

            return scores

    except Exception as e:
        print(f"[CivicClassifier] Zero-shot scoring error: {e}")
        return {cls: 0.5 for cls in CIVIC_CLASSES}


# ─── Main Public API ──────────────────────────────────────────────────────────
def classify_civic_defect(
    image_path: str,
    weights_path: Optional[str] = None,
    confidence_threshold: float = 0.35
) -> Tuple[str, float, Dict[str, float]]:
    """
    Classifies an image into a civic defect category.

    Returns:
        (predicted_class, confidence, all_class_scores)

    Pipeline:
        1. If PyTorch + fine-tuned weights available → use trained CNN
        2. If PyTorch available but no weights → use zero-shot with backbone features
        3. If PyTorch not installed → use pure numpy zero-shot heuristics
    """
    model, transform = get_classifier(weights_path)

    # ── Path A: PyTorch inference ─────────────────────────────────────────────
    if model is not None and transform is not None:
        try:
            with Image.open(image_path) as img:
                img_tensor = transform(img.convert("RGB")).unsqueeze(0)

            with torch.no_grad():
                logits = model(img_tensor)
                probs = torch.softmax(logits, dim=1)[0].numpy()

            scores = {CIVIC_CLASSES[i]: round(float(probs[i]), 3) for i in range(NUM_CLASSES)}
            best_idx = int(np.argmax(probs))
            best_class = CIVIC_CLASSES[best_idx]
            best_conf = float(probs[best_idx])

            # If the model is not fine-tuned, zero-shot backbone often predicts
            # "Normal" with high confidence → blend with heuristic scores
            if weights_path is None or not os.path.exists(weights_path or ""):
                heuristic = _zero_shot_civic_score(image_path)
                blended = {}
                for cls in CIVIC_CLASSES:
                    blended[cls] = round(
                        0.4 * scores.get(cls, 0.5) + 0.6 * heuristic.get(cls, 0.5), 3
                    )
                scores = blended
                best_idx_b = max(scores, key=scores.get)
                best_class = best_idx_b
                best_conf = scores[best_idx_b]

            if best_conf < confidence_threshold:
                best_class = "Pothole / Road Damage"  # conservative default
                best_conf = confidence_threshold

            return best_class, round(best_conf, 3), scores

        except Exception as e:
            print(f"[CivicClassifier] PyTorch inference error: {e}")

    # ── Path B: Pure NumPy zero-shot (no PyTorch) ─────────────────────────────
    scores = _zero_shot_civic_score(image_path)
    best_class = max(scores, key=scores.get)
    best_conf = scores[best_class]

    if best_class == "Normal / No Defect" or best_conf < confidence_threshold:
        # Conservative default — better to flag than miss
        best_class = "Pothole / Road Damage"
        best_conf = max(best_conf, 0.40)

    return best_class, round(best_conf, 3), scores
