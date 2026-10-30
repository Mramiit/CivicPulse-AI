from fastapi import APIRouter
from backend.app.api.v1.reports import router as reports_router
from backend.app.api.v1.tickets import router as tickets_router
from backend.app.api.v1.analytics import router as analytics_router

api_router = APIRouter()
api_router.include_router(reports_router, tags=["Reports"])
api_router.include_router(tickets_router, tags=["Tickets"])
api_router.include_router(analytics_router, tags=["Analytics"])
