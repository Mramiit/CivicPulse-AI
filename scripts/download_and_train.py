"""
CivicPulse AI — Civic Defect YOLOv8 Training Script
======================================================
Downloads the RDD2022 (Road Damage Detection) dataset and fine-tunes
YOLOv8n on it so the model actually recognizes:
  - Potholes
  - Longitudinal Cracks
  - Transverse Cracks
  - Alligator Cracks

Usage:
    python scripts/download_and_train.py

Requirements:
    pip install ultralytics roboflow

The trained weights will be saved to:
    weights/civicpulse_yolo.pt
"""

import os
import sys
import shutil

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(ROOT_DIR, "weights")
DATA_DIR    = os.path.join(ROOT_DIR, "data", "rdd2022")
FINAL_WEIGHTS = os.path.join(WEIGHTS_DIR, "civicpulse_yolo.pt")

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ─── Step 1: Download RDD2022 via Roboflow (free, no login needed) ────────────
def download_dataset_roboflow():
    """
    Downloads the RDD2022 dataset in YOLOv8 format from Roboflow Universe.
    The dataset has ~47,000 road damage images across 4 classes.
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[ERROR] roboflow not installed. Run: pip install roboflow")
        sys.exit(1)

    print("[INFO] Downloading RDD2022 dataset from Roboflow Universe...")
    rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")   # ← Get free key at roboflow.com
    project = rf.workspace("roboflow-universe-projects").project("road-damage-detection-uxxsp")
    dataset = project.version(4).download("yolov8", location=DATA_DIR)
    print(f"[INFO] Dataset downloaded to: {DATA_DIR}")
    return dataset.location


def download_dataset_direct():
    """
    Alternative: Download RDD2022 directly from the official GitHub release
    (no API key required). Uses the Japan subset (~10k images).
    """
    import urllib.request
    import zipfile

    URL = "https://github.com/sekilab/RoadDamageDetector/releases/download/dataset3.0/RoadDamageDataset_Japan.zip"
    zip_path = os.path.join(DATA_DIR, "rdd2022_japan.zip")

    print("[INFO] Downloading RDD2022 Japan subset...")
    urllib.request.urlretrieve(URL, zip_path)
    print("[INFO] Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)
    os.remove(zip_path)
    print(f"[INFO] Dataset ready at: {DATA_DIR}")


# ─── Step 2: Create dataset YAML ──────────────────────────────────────────────
YAML_CONTENT = """
# RDD2022 — Road Damage Detection Dataset
# 4 civic defect classes

path: {data_dir}
train: train/images
val:   valid/images
test:  test/images

nc: 4
names:
  0: Longitudinal Crack
  1: Transverse Crack
  2: Alligator Crack
  3: Pothole
"""

def write_dataset_yaml(data_dir: str) -> str:
    yaml_path = os.path.join(data_dir, "civicpulse_rdd.yaml")
    with open(yaml_path, "w") as f:
        f.write(YAML_CONTENT.format(data_dir=data_dir))
    print(f"[INFO] Dataset YAML written: {yaml_path}")
    return yaml_path


# ─── Step 3: Fine-tune YOLOv8 ────────────────────────────────────────────────
def train_yolo(yaml_path: str):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    print("[INFO] Loading base YOLOv8n pretrained weights...")
    model = YOLO("yolov8n.pt")  # Auto-downloads ~6MB pretrained COCO weights

    print("[INFO] Starting fine-tuning on civic defect dataset...")
    results = model.train(
        data=yaml_path,
        epochs=50,          # Increase to 100+ for higher accuracy
        imgsz=640,          # Standard YOLO input size
        batch=16,           # Reduce to 8 if GPU memory < 6GB
        patience=10,        # Early stopping
        device=0,           # GPU (use 'cpu' if no CUDA GPU)
        project=WEIGHTS_DIR,
        name="civicpulse_run",
        exist_ok=True,
        verbose=True,

        # Augmentation — helps generalize across different road types
        flipud=0.3,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        degrees=10,
        translate=0.1,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )

    # Copy best weights to canonical path
    best_weights = os.path.join(WEIGHTS_DIR, "civicpulse_run", "weights", "best.pt")
    if os.path.exists(best_weights):
        shutil.copy(best_weights, FINAL_WEIGHTS)
        print(f"\n[SUCCESS] Training complete!")
        print(f"[SUCCESS] Civic defect model saved → {FINAL_WEIGHTS}")
        print(f"[SUCCESS] Update your .env: YOLO_MODEL_PATH={FINAL_WEIGHTS}")
    else:
        print(f"[WARNING] Could not find best.pt. Check {WEIGHTS_DIR}/civicpulse_run/")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  CivicPulse AI — Civic Defect YOLOv8 Training")
    print("=" * 60)

    # Option A: Use Roboflow (requires free API key)
    # data_dir = download_dataset_roboflow()

    # Option B: Direct download (no API key)
    # download_dataset_direct()
    # data_dir = DATA_DIR

    # If you already have the dataset, set data_dir manually:
    data_dir = DATA_DIR  # ← point to your dataset folder

    yaml_path = write_dataset_yaml(data_dir)
    train_yolo(yaml_path)
