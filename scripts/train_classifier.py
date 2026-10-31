"""
CivicPulse AI — MobileNetV2 Civic Defect Classifier Fine-Tuning Script
=======================================================================
Fine-tunes the MobileNetV2 head on your own labeled dataset of civic defect images.

Dataset folder structure expected:
    data/classifier/
        Pothole/              ← folders named exactly as CIVIC_CLASSES
            img001.jpg
            img002.jpg
        Longitudinal Crack/
            img001.jpg
        Transverse Crack/
        Alligator Crack/
        Waterlogging & Drainage/
        Solid Waste & Garbage/
        Open Manhole/
        Streetlight Hazard/
        Normal/

Quick dataset sources (download & organize manually):
  - Pothole images:       https://www.kaggle.com/datasets/atulyakumar98/pothole-detection-dataset
  - Road damage:          https://github.com/sekilab/RoadDamageDetector
  - Garbage:              https://www.kaggle.com/datasets/techsash/waste-classification-data
  - Waterlogging:         https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia (NO - wrong)
  - General civic:        Collect manually using Google Images / Mapillary

Usage:
    pip install torch torchvision
    python scripts/train_classifier.py

Output:
    weights/civicpulse_classifier.pt
"""

import os
import sys
import time
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from torchvision import datasets, transforms, models
except ImportError:
    print("[ERROR] PyTorch not installed.")
    print("  Run: pip install torch torchvision")
    sys.exit(1)

# ─── Config ───────────────────────────────────────────────────────────────────
ROOT_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR       = os.path.join(ROOT_DIR, "data", "classifier")
WEIGHTS_DIR    = os.path.join(ROOT_DIR, "weights")
OUTPUT_WEIGHTS = os.path.join(WEIGHTS_DIR, "civicpulse_classifier.pt")

EPOCHS         = 30
BATCH_SIZE     = 32
LEARNING_RATE  = 1e-3
FREEZE_EPOCHS  = 5       # Epochs to train only head; then unfreeze backbone
IMG_SIZE       = 224
NUM_WORKERS    = 0       # Set to 4 on Linux for faster data loading

# ─── Transforms ───────────────────────────────────────────────────────────────
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    # Freeze backbone initially
    for param in model.features.parameters():
        param.requires_grad = False
    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    return model


def make_weighted_sampler(dataset):
    """Handles class imbalance — rare classes get upweighted."""
    class_counts = np.bincount(dataset.targets)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[t] for t in dataset.targets]
    return WeightedRandomSampler(sample_weights, len(sample_weights))


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    if not os.path.isdir(DATA_DIR):
        print(f"[ERROR] Dataset folder not found: {DATA_DIR}")
        print("  Create the folder structure described at the top of this script.")
        sys.exit(1)

    full_dataset = datasets.ImageFolder(DATA_DIR, transform=TRAIN_TRANSFORM)
    num_classes  = len(full_dataset.classes)
    print(f"[INFO] Classes found ({num_classes}): {full_dataset.classes}")
    print(f"[INFO] Total images: {len(full_dataset)}")

    # 80/20 train/val split
    n_val   = max(1, int(len(full_dataset) * 0.20))
    n_train = len(full_dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    val_set.dataset.transform = VAL_TRANSFORM

    sampler    = make_weighted_sampler(full_dataset)
    train_idx  = train_set.indices
    train_weights = [sampler.weights[i] for i in train_idx]
    train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE,
        sampler=train_sampler, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )

    best_val_acc  = 0.0
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    print("\n[INFO] Starting training...")
    for epoch in range(1, EPOCHS + 1):

        # Unfreeze backbone after FREEZE_EPOCHS
        if epoch == FREEZE_EPOCHS + 1:
            print(f"[INFO] Epoch {epoch}: Unfreezing MobileNetV2 backbone for full fine-tuning")
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE * 0.1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=EPOCHS - FREEZE_EPOCHS, eta_min=1e-6
            )

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * imgs.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total   += imgs.size(0)

        scheduler.step()

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total   += imgs.size(0)

        train_acc = 100.0 * train_correct / train_total
        val_acc   = 100.0 * val_correct   / val_total
        avg_loss  = train_loss / train_total

        print(
            f"  Epoch [{epoch:02d}/{EPOCHS}]  "
            f"Loss: {avg_loss:.4f}  "
            f"Train Acc: {train_acc:.1f}%  "
            f"Val Acc: {val_acc:.1f}%"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUTPUT_WEIGHTS)
            print(f"  ✓ Saved best model (Val Acc: {best_val_acc:.1f}%)")

    print(f"\n[SUCCESS] Training complete. Best Val Accuracy: {best_val_acc:.1f}%")
    print(f"[SUCCESS] Weights saved → {OUTPUT_WEIGHTS}")
    print(f"[SUCCESS] Update your .env: CLASSIFIER_WEIGHTS_PATH={OUTPUT_WEIGHTS}")


if __name__ == "__main__":
    train()
