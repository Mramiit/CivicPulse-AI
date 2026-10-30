import os
import tempfile
import numpy as np
from PIL import Image

from backend.app.core.severity import calculate_multimodal_severity, get_hazard_weight
from backend.app.core.forensics import compute_dhash, hamming_distance, inspect_image_forensics
from backend.app.services.vision_service import run_vision_pipeline

def test_multimodal_severity_calculation():
    # Test high severity case: high conf, high area, critical hazard (manhole), high rain
    severity, breakdown = calculate_multimodal_severity(
        c_conf=0.90,
        r_area=0.40,
        category="Open Manhole / Drain",
        e_weather=0.80
    )
    assert 0.0 <= severity <= 1.0
    assert severity >= 0.70
    assert breakdown["risk_level"] == "CRITICAL"
    assert "formula" in breakdown

def test_multimodal_severity_low_case():
    severity, breakdown = calculate_multimodal_severity(
        c_conf=0.20,
        r_area=0.05,
        category="Solid Waste & Garbage",
        e_weather=0.05
    )
    assert 0.0 <= severity <= 1.0
    assert severity < 0.40
    assert breakdown["risk_level"] == "LOW"

def test_forensics_dhash_and_hamming():
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f1, \
         tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f2:
        
        # Create dummy image 1
        img1 = Image.new("RGB", (200, 200), color=(128, 50, 50))
        img1.save(f1.name)
        
        # Create identical image 2
        img2 = Image.new("RGB", (200, 200), color=(128, 50, 50))
        img2.save(f2.name)

        hash1 = compute_dhash(f1.name)
        hash2 = compute_dhash(f2.name)
        
        assert len(hash1) == 16
        assert len(hash2) == 16
        assert hamming_distance(hash1, hash2) == 0

        trust_score, meta = inspect_image_forensics(f1.name)
        assert 0.0 <= trust_score <= 1.0
        assert "dimensions" in meta

    os.remove(f1.name)
    os.remove(f2.name)

def test_vision_pipeline_execution():
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f, \
         tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as out_f:
        
        # Create test image with contrasting rectangle representing road defect
        img = Image.new("RGB", (320, 240), color=(100, 100, 100))
        for x in range(120, 200):
            for y in range(80, 160):
                img.putpixel((x, y), (20, 20, 20))
        img.save(f.name)

        detections, c_conf, r_area, preview = run_vision_pipeline(f.name, out_f.name)
        
        assert 0.0 <= c_conf <= 1.0
        assert 0.0 <= r_area <= 1.0
        assert os.path.exists(out_f.name)

    os.remove(f.name)
    os.remove(out_f.name)
