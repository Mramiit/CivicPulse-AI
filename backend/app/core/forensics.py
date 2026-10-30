import os
from typing import Tuple, Dict, Any, Optional
from PIL import Image, ExifTags
import numpy as np

def compute_dhash(image_path: str, hash_size: int = 8) -> str:
    """
    Computes a 64-bit difference hash (dHash) for perceptual similarity comparison.
    Resizes image to (hash_size + 1, hash_size), converts to grayscale,
    and computes pixel intensity gradients horizontally.
    """
    try:
        with Image.open(image_path) as img:
            # Convert to grayscale and resize to (9, 8)
            img_gray = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            pixels = np.array(img_gray, dtype=np.float32)
            
            # Compare adjacent pixels along each row
            difference = pixels[:, 1:] > pixels[:, :-1]
            
            # Convert boolean array to 64-bit integer
            decimal_value = 0
            for idx, val in enumerate(difference.flatten()):
                if val:
                    decimal_value |= (1 << idx)
            
            # Format as 16-character hex string (64 bits)
            return f"{decimal_value:016x}"
    except Exception as e:
        # Fallback pseudo-hash based on file size and basic stats
        file_size = os.path.getsize(image_path) if os.path.exists(image_path) else 0
        return f"{file_size:016x}"

def hamming_distance(hash1: str, hash2: str) -> int:
    """
    Computes Hamming distance between two 16-character hex hashes (64-bit).
    Distance <= 5 indicates near-duplicate or identical image.
    """
    try:
        val1 = int(hash1, 16)
        val2 = int(hash2, 16)
        xor_val = val1 ^ val2
        return bin(xor_val).count("1")
    except Exception:
        return 64

def inspect_image_forensics(image_path: str) -> Tuple[float, Dict[str, Any]]:
    """
    Inspects image EXIF metadata and structural properties to compute a trust_score in [0.0, 1.0].
    - Authentic device camera capture with timestamp/camera metadata: 0.85 - 1.0
    - Web / canvas captured / modern mobile stripped EXIF: 0.70 - 0.85
    - Suspiciously small / truncated files: 0.30 - 0.50
    """
    metadata: Dict[str, Any] = {
        "has_exif": False,
        "camera_make": None,
        "camera_model": None,
        "software": None,
        "datetime_original": None,
        "dimensions": None,
        "is_camera_direct": False
    }
    
    trust_score = 0.75  # Neutral baseline for modern browser uploads
    
    try:
        with Image.open(image_path) as img:
            metadata["dimensions"] = f"{img.width}x{img.height}"
            
            # Resolution sanity check
            if img.width < 100 or img.height < 100:
                trust_score -= 0.35
            elif img.width >= 640 and img.height >= 480:
                trust_score += 0.05
                
            exif_data = img._getexif()
            if exif_data:
                metadata["has_exif"] = True
                trust_score += 0.15
                
                for tag_id, value in exif_data.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    if tag_name == "Make":
                        metadata["camera_make"] = str(value).strip()
                    elif tag_name == "Model":
                        metadata["camera_model"] = str(value).strip()
                    elif tag_name == "Software":
                        metadata["software"] = str(value).strip()
                    elif tag_name == "DateTimeOriginal":
                        metadata["datetime_original"] = str(value).strip()
                
                if metadata["camera_make"] or metadata["camera_model"]:
                    metadata["is_camera_direct"] = True
                    trust_score += 0.05
    except Exception as e:
        metadata["error"] = str(e)
        trust_score = 0.50

    trust_score = float(np.clip(trust_score, 0.1, 1.0))
    return round(trust_score, 2), metadata
