import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.app.config import settings
from backend.app.db.session import engine
from backend.app.db.base import Base
import backend.app.models.ticket  # Registers models on Base.metadata
from backend.app.api.v1 import api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure all database tables exist on startup
    Base.metadata.create_all(bind=engine)
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

# Mount uploads directory for static image serving
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

# Include API v1 Router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Serve Frontend HTML
FRONTEND_FILE = os.path.join(settings.BASE_DIR, "frontend", "index.html")

@app.get("/", include_in_schema=False)
async def serve_dashboard():
    if os.path.exists(FRONTEND_FILE):
        return FileResponse(FRONTEND_FILE)
    return {"message": "CivicPulse AI Engine active. API docs available at /docs"}

@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "online",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION
    }
