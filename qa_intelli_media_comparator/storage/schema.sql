CREATE TABLE IF NOT EXISTS reports (
    report_id       TEXT PRIMARY KEY,
    timestamp       TEXT NOT NULL,
    media_type      TEXT NOT NULL,
    dut_file        TEXT NOT NULL,
    reference_file  TEXT,
    overall_grade   TEXT NOT NULL,
    processing_time_ms INTEGER,
    json_path       TEXT NOT NULL,
    annotated_path  TEXT,
    diff_path       TEXT
);

CREATE INDEX IF NOT EXISTS idx_reports_timestamp ON reports(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_reports_grade      ON reports(overall_grade);
CREATE INDEX IF NOT EXISTS idx_reports_media_type ON reports(media_type);
