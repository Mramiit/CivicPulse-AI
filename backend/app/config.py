import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    PROJECT_NAME: str = "CivicPulse AI"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Server Host & Port
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./civicpulse.db")
    
    # Upload Directories
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    UPLOAD_DIR: str = os.path.join(BASE_DIR, "uploads")
    WEIGHTS_DIR: str = os.path.join(BASE_DIR, "weights")
    
    # Model Weights
    YOLO_MODEL_PATH: str = os.path.join(WEIGHTS_DIR, "yolov8n.pt")
    # Civic fine-tuned YOLO weights (run: python scripts/download_and_train.py)
    CIVIC_YOLO_PATH: str = os.path.join(WEIGHTS_DIR, "civicpulse_yolo.pt")
    # MobileNetV2 classifier weights (run: python scripts/train_classifier.py)
    CLASSIFIER_WEIGHTS_PATH: str = os.path.join(WEIGHTS_DIR, "civicpulse_classifier.pt")
    WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "base")
    
    # Live External APIs
    NOMINATIM_URL: str = "https://nominatim.openstreetmap.org/reverse"
    NOMINATIM_USER_AGENT: str = "CivicPulseAI-Engine/1.0 (civicpulse-contact@gov.local)"
    OPEN_METEO_URL: str = "https://api.open-meteo.com/v1/forecast"
    
    # Deduplication & Clustering parameters
    DBSCAN_EPSILON_METERS: float = 35.0  # 35 meters Haversine radius
    DBSCAN_MIN_SAMPLES: int = 1
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["*"]

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.WEIGHTS_DIR, exist_ok=True)
