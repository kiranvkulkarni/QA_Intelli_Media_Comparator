from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from ..models.report import ComparisonReport


class ReportStore:
    """Persists ComparisonReport to disk (JSON) and maintains a SQLite index."""

    def __init__(self, reports_dir: Path) -> None:
        self.reports_dir = reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = reports_dir / "index.db"
        self._init_db()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        schema_path = Path(__file__).parent / "schema.sql"
        ddl = schema_path.read_text(encoding="utf-8")
        with self._connect() as conn:
            conn.executescript(ddl)

    # ── Public API ─────────────────────────────────────────────────────────────

    def save(self, report: ComparisonReport) -> Path:
        """Serialize report to JSON and index in SQLite. Returns JSON file path."""
        json_path = self.reports_dir / f"{report.report_id}.json"
        json_path.write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO reports
                    (report_id, timestamp, media_type, dut_file, reference_file,
                     overall_grade, processing_time_ms, json_path, annotated_path, diff_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.report_id,
                    report.timestamp.isoformat(),
                    report.media_type.value,
                    report.dut_file,
                    report.reference_file,
                    report.overall_grade.value,
                    report.processing_time_ms,
                    str(json_path),
                    report.annotated_image_path,
                    report.diff_image_path,
                ),
            )
        return json_path

    def load(self, report_id: str) -> Optional[ComparisonReport]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT json_path FROM reports WHERE report_id = ?", (report_id,)
            ).fetchone()
        if row is None:
            return None
        json_path = Path(row["json_path"])
        if not json_path.exists():
            return None
        return ComparisonReport.model_validate_json(json_path.read_text(encoding="utf-8"))

    def list_reports(
        self,
        limit: int = 20,
        offset: int = 0,
        grade: Optional[str] = None,
        media_type: Optional[str] = None,
    ) -> list[dict]:
        clauses = []
        params: list = []
        if grade:
            clauses.append("overall_grade = ?")
            params.append(grade)
        if media_type:
            clauses.append("media_type = ?")
            params.append(media_type)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.extend([limit, offset])
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT report_id, timestamp, media_type, dut_file, reference_file,
                       overall_grade, processing_time_ms, annotated_path
                FROM reports
                {where}
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def get_annotated_path(self, report_id: str) -> Optional[Path]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT annotated_path FROM reports WHERE report_id = ?", (report_id,)
            ).fetchone()
        if row and row["annotated_path"]:
            p = Path(row["annotated_path"])
            return p if p.exists() else None
        return None

    def get_diff_path(self, report_id: str) -> Optional[Path]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT diff_path FROM reports WHERE report_id = ?", (report_id,)
            ).fetchone()
        if row and row["diff_path"]:
            p = Path(row["diff_path"])
            return p if p.exists() else None
        return None
