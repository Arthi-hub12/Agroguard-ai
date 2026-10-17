"""SQLite persistence. Data survives restarts; duplicates are prevented by keys."""
from __future__ import annotations

import json
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    location TEXT,
    crop TEXT,
    temperature REAL, humidity REAL, rainfall REAL, wind_speed REAL, pressure REAL,
    pest_observed INTEGER,
    pest_count REAL,
    source TEXT NOT NULL DEFAULT 'unknown',   -- real | replay | synthetic
    data_quality_warnings TEXT,
    processed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_loc ON events(location);

CREATE TABLE IF NOT EXISTS anomaly_results (
    event_id TEXT PRIMARY KEY REFERENCES events(event_id),
    score REAL, is_anomaly INTEGER, severity TEXT,
    top_features TEXT, model_version TEXT, created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS risk_predictions (
    event_id TEXT PRIMARY KEY REFERENCES events(event_id),
    probability REAL, level TEXT, model_version TEXT,
    validated INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL REFERENCES events(event_id),
    timestamp TEXT NOT NULL, location TEXT,
    category TEXT NOT NULL, severity TEXT NOT NULL,
    evidence TEXT, trigger_rule TEXT, data_quality_warnings TEXT,
    recommendation TEXT, source TEXT,
    status TEXT NOT NULL DEFAULT 'open',      -- open | acknowledged | resolved
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_cat ON alerts(category);

CREATE TABLE IF NOT EXISTS model_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT, model_version TEXT, trained_at TEXT, data_origin TEXT,
    summary_json TEXT, recorded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS processing_errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT, error TEXT, raw TEXT, created_at TEXT NOT NULL
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Optional[Path | str] = None) -> None:
        self.path = Path(path) if path else Path(settings.DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self.connect() as c:
            c.executescript(SCHEMA)

    # ---------------------------------------------------------------- writes
    def event_exists(self, event_id: str) -> bool:
        with self.connect() as c:
            return c.execute("SELECT 1 FROM events WHERE event_id=?", (event_id,)).fetchone() is not None

    def insert_event(self, ev: dict) -> bool:
        """Returns True if inserted, False if it was a duplicate."""
        with self.connect() as c:
            cur = c.execute(
                """INSERT OR IGNORE INTO events (event_id,timestamp,location,crop,temperature,humidity,
                   rainfall,wind_speed,pressure,pest_observed,pest_count,source,data_quality_warnings,processed_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (ev["event_id"], ev["timestamp"], ev.get("location"), ev.get("crop"),
                 ev.get("temperature"), ev.get("humidity"), ev.get("rainfall"), ev.get("wind_speed"),
                 ev.get("pressure"), ev.get("pest_observed"), ev.get("pest_count"),
                 ev.get("source", "unknown"), json.dumps(ev.get("warnings", [])), now_iso()))
            return cur.rowcount == 1

    def insert_anomaly(self, event_id: str, score: float, is_anomaly: bool, severity: str,
                       top_features: list, model_version: str) -> None:
        with self.connect() as c:
            c.execute("""INSERT OR REPLACE INTO anomaly_results VALUES (?,?,?,?,?,?,?)""",
                      (event_id, score, int(is_anomaly), severity, json.dumps(top_features), model_version, now_iso()))

    def insert_risk(self, event_id: str, probability: float, level: str, model_version: str, validated: bool) -> None:
        with self.connect() as c:
            c.execute("""INSERT OR REPLACE INTO risk_predictions VALUES (?,?,?,?,?,?)""",
                      (event_id, probability, level, model_version, int(validated), now_iso()))

    def insert_alert(self, alert: dict) -> bool:
        ts = now_iso()
        with self.connect() as c:
            cur = c.execute(
                """INSERT OR IGNORE INTO alerts (alert_id,event_id,timestamp,location,category,severity,evidence,
                   trigger_rule,data_quality_warnings,recommendation,source,status,created_at,updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?, 'open', ?, ?)""",
                (alert["alert_id"], alert["event_id"], alert["timestamp"], alert.get("location"),
                 alert["category"], alert["severity"], json.dumps(alert.get("evidence", {}), default=str),
                 alert.get("trigger_rule"), json.dumps(alert.get("warnings", [])),
                 alert.get("recommendation"), alert.get("source"), ts, ts))
            return cur.rowcount == 1

    def update_alert_status(self, alert_id: str, status: str) -> bool:
        if status not in ("open", "acknowledged", "resolved"):
            raise ValueError("status must be open, acknowledged or resolved")
        with self.connect() as c:
            cur = c.execute("UPDATE alerts SET status=?, updated_at=? WHERE alert_id=?", (status, now_iso(), alert_id))
            return cur.rowcount == 1

    def record_model_run(self, name: str, meta: dict) -> None:
        with self.connect() as c:
            c.execute("""INSERT INTO model_runs (model_name,model_version,trained_at,data_origin,summary_json,recorded_at)
                         VALUES (?,?,?,?,?,?)""",
                      (name, meta.get("model_version"), meta.get("trained_at"), meta.get("data_origin"),
                       json.dumps(meta, default=str), now_iso()))

    def log_error(self, event_id: Optional[str], error: str, raw: object = None) -> None:
        with self.connect() as c:
            c.execute("INSERT INTO processing_errors (event_id,error,raw,created_at) VALUES (?,?,?,?)",
                      (event_id, error, json.dumps(raw, default=str) if raw is not None else None, now_iso()))

    # ----------------------------------------------------------------- reads
    def query_df(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        with self.connect() as c:
            return pd.read_sql_query(sql, c, params=params)

    def counts(self) -> dict:
        with self.connect() as c:
            g = lambda q: c.execute(q).fetchone()[0]  # noqa: E731
            return {
                "events": g("SELECT COUNT(*) FROM events"),
                "synthetic_events": g("SELECT COUNT(*) FROM events WHERE source='synthetic'"),
                "anomalies": g("SELECT COUNT(*) FROM anomaly_results WHERE is_anomaly=1"),
                "alerts_open": g("SELECT COUNT(*) FROM alerts WHERE status='open'"),
                "alerts_total": g("SELECT COUNT(*) FROM alerts"),
                "risk_predictions": g("SELECT COUNT(*) FROM risk_predictions"),
                "errors": g("SELECT COUNT(*) FROM processing_errors"),
            }

    def backup(self, dest_dir: Optional[Path | str] = None) -> Path:
        """Consistent online backup using SQLite's backup API."""
        dest_dir = Path(dest_dir) if dest_dir else self.path.parent / "backups"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"agroguard_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.db"
        src = sqlite3.connect(self.path)
        dst = sqlite3.connect(dest)
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()
        return dest

    def reset(self) -> None:
        """Deletes all rows. Only used by explicit user action / tests."""
        with self.connect() as c:
            for t in ("alerts", "risk_predictions", "anomaly_results", "events", "processing_errors"):
                c.execute(f"DELETE FROM {t}")


def init_db_cli() -> int:
    db = Database()
    print(f"Database ready at {db.path}")
    print(db.counts())
    return 0


if __name__ == "__main__":
    raise SystemExit(init_db_cli())
