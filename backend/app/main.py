import asyncio
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy import text
from starlette.concurrency import run_in_threadpool

from backend.app.config import settings
from backend.app.db.session import engine
from backend.app.db.base import Base
import backend.app.models.ticket  # Registers models on Base.metadata
from backend.app.api.v1 import api_router


def _ensure_schema_columns(bind_engine):
    """Safely adds newly introduced columns if running against an existing SQLite DB."""
    if not str(bind_engine.url).startswith("sqlite"):
        return

    new_columns = [
        ("tickets", "video_path", "VARCHAR(255)"),
        ("tickets", "verification_video_path", "VARCHAR(255)"),
        ("tickets", "media_type", "VARCHAR(10) DEFAULT 'image'"),
        ("tickets", "video_analysis_details", "TEXT"),
        ("citizen_reports", "video_path", "VARCHAR(255)"),
    ]

    try:
        with bind_engine.connect() as conn:
            for table, col, col_type in new_columns:
                try:
                    res = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
                    existing_cols = [row[1] for row in res]
                    if col not in existing_cols:
                        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}"))
                        conn.commit()
                except Exception:
                    pass
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure all database tables exist on startup (< 0.1s)
    Base.metadata.create_all(bind=engine)
    _ensure_schema_columns(engine)

    # Pre-warm vision models in background task (non-blocking)
    # This allows Uvicorn to bind and open the port in < 0.5s so Render's port scanner succeeds!
    async def _bg_warmup():
        try:
            from backend.app.services.vision_service import get_yolo_model
            await run_in_threadpool(get_yolo_model)
            from backend.app.services.classifier_service import classify_civic_defect
            print("[Startup] Vision models and classifier pre-warmed successfully in background.")
        except Exception as e:
            print(f"[Startup] Vision background warm-up notice: {e}")

    asyncio.create_task(_bg_warmup())

    # Immediately yield so Uvicorn binds to $PORT in under 1 second!
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Autonomous Multimodal Civic Grievance & Infrastructure Triage Engine",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount uploads directory for static media serving (images + videos)
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

# Include API v1 Router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Serve Frontend HTML
FRONTEND_FILE = os.path.join(settings.BASE_DIR, "frontend", "index.html")

@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
async def serve_dashboard():
    if os.path.exists(FRONTEND_FILE):
        return FileResponse(FRONTEND_FILE)
    return {"message": "CivicPulse AI Engine active. API docs available at /docs"}

@app.api_route("/health", methods=["GET", "HEAD"], tags=["System"])
def health_check():
    return {
        "status": "online",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION
    }
