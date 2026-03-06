from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ..config import get_settings
from ..services.pipeline import ComparisonPipeline
from ..storage.report_store import ReportStore


@lru_cache(maxsize=1)
def get_pipeline() -> ComparisonPipeline:
    return ComparisonPipeline()


@lru_cache(maxsize=1)
def get_report_store() -> ReportStore:
    settings = get_settings()
    return ReportStore(settings.reports_dir)
