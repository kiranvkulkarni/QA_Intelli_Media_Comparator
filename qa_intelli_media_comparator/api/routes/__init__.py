from .compare import router as compare_router
from .analyze import router as analyze_router
from .health import router as health_router
from .reports import router as reports_router

__all__ = ["compare_router", "analyze_router", "health_router", "reports_router"]
