"""Event pipeline: validate -> dedupe -> features -> score -> alert -> persist.

This processes events from files, a replay of historical records, or SYNTHETIC
demo events. It is an event-driven processing demo, not a real-time sensor
network or a production streaming platform.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

from config import settings
from src import alert_engine
from src.database import Database
from src.inference import ModelNotAvailable, Scorer

ALLOWED_SOURCES = ("real", "replay", "synthetic", "unknown")


class ValidationError(ValueError):
    pass


def validate_event(raw: dict) -> dict:
    """Return a cleaned event dict or raise ValidationError with a helpful message."""
    if not isinstance(raw, dict):
        raise ValidationError("Event must be a dictionary/JSON object.")
    missing = [f for f in settings.EVENT_REQUIRED_FIELDS if raw.get(f) in (None, "")]
    if missing:
        raise ValidationError(f"Missing required field(s): {missing}. Required: {list(settings.EVENT_REQUIRED_FIELDS)}")

    ts = pd.to_datetime(raw["timestamp"], errors="coerce")
    if pd.isna(ts):
        raise ValidationError(f"Invalid timestamp: {raw['timestamp']!r}. Use e.g. 2024-06-01 or 2024-06-01T12:00:00.")

    ev: dict = {"event_id": str(raw["event_id"]).strip(), "timestamp": ts.isoformat()}
    warnings: list = []

    loc = raw.get("location")
    if loc in (None, "") or (isinstance(loc, float) and np.isnan(loc)):
        warnings.append("Location missing; per-location context unavailable.")
        ev["location"] = None
    else:
        ev["location"] = str(loc).strip().title()
    crop = raw.get("crop")
    ev["crop"] = None if crop in (None, "") or (isinstance(crop, float) and np.isnan(crop)) else str(crop).strip().lower()

    ranges = settings.CLEANING.valid_ranges
    key_for = {"temperature": "temperature", "humidity": "humidity", "rainfall": "rainfall",
               "wind_speed": "wind_speed", "pressure": "pressure"}
    n_weather = 0
    for f in settings.EVENT_WEATHER_FIELDS:
        v = raw.get(f)
        if v in (None, "") or (isinstance(v, float) and np.isnan(v)):
            ev[f] = None
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            warnings.append(f"{f} is not numeric ({v!r}); ignored.")
            ev[f] = None
            continue
        lo, hi = ranges[key_for[f]]
        if not (lo <= fv <= hi):
            warnings.append(f"{f}={fv} outside plausible range [{lo}, {hi}]; ignored.")
            ev[f] = None
            continue
        ev[f] = fv
        n_weather += 1
    if n_weather == 0:
        raise ValidationError("Event has no usable weather measurements; nothing to analyse.")
    if n_weather < 3:
        warnings.append(f"Only {n_weather} weather measurement(s) available; score is less reliable.")

    po = raw.get("pest_observed")
    ev["pest_observed"] = None if po in (None, "") else int(str(po).strip().lower() in ("1", "true", "yes", "y"))
    pc = raw.get("pest_count")
    try:
        ev["pest_count"] = None if pc in (None, "") else float(pc)
    except (TypeError, ValueError):
        ev["pest_count"] = None
        warnings.append("pest_count is not numeric; ignored.")

    src = str(raw.get("source", "unknown")).lower()
    ev["source"] = src if src in ALLOWED_SOURCES else "unknown"
    ev["warnings"] = warnings
    return ev


class EventProcessor:
    """Stateful processor. Keeps a short per-location history for rolling features."""

    HISTORY = 7

    def __init__(self, db: Optional[Database] = None, scorer: Optional[Scorer] = None) -> None:
        self.db = db or Database()
        self.scorer = scorer or Scorer()
        self._history: dict = defaultdict(lambda: deque(maxlen=self.HISTORY))
        self._warm_history()

    def _warm_history(self) -> None:
        """Rebuild rolling context from stored events so restarts keep working."""
        try:
            df = self.db.query_df("SELECT * FROM events ORDER BY timestamp")
        except Exception:  # noqa: BLE001
            return
        for rec in df.to_dict("records"):
            self._history[rec.get("location") or "_none_"].append(rec)

    def _features(self, ev: dict) -> dict:
        key = ev.get("location") or "_none_"
        hist = list(self._history[key])
        row = {f: ev.get(f) for f in settings.EVENT_WEATHER_FIELDS}
        ts = pd.Timestamp(ev["timestamp"])
        doy = ts.dayofyear
        row["season_sin"] = float(np.sin(2 * np.pi * doy / 366.0))
        row["season_cos"] = float(np.cos(2 * np.pi * doy / 366.0))
        # past-only windows (history excludes the current event)
        for f in settings.EVENT_WEATHER_FIELDS:
            vals = [h[f] for h in hist if h.get(f) is not None and not pd.isna(h.get(f))]
            for w in (3, 7):
                recent = vals[-w:]
                row[f"{f}_roll{w}_mean"] = float(np.mean(recent)) if recent else np.nan
                if f == "rainfall":
                    row[f"rainfall_roll{w}_sum"] = float(np.sum(recent)) if recent else np.nan
            row[f"{f}_change"] = (ev[f] - vals[-1]) if (ev.get(f) is not None and vals) else np.nan
        return row

    def process_event(self, raw: dict) -> dict:
        """Process one event. Always returns a result dict; never raises on bad input."""
        eid = str(raw.get("event_id", "")) if isinstance(raw, dict) else ""
        try:
            ev = validate_event(raw)
        except ValidationError as exc:
            self.db.log_error(eid or None, str(exc), raw)
            return {"status": "rejected", "event_id": eid, "error": str(exc)}

        if self.db.event_exists(ev["event_id"]):
            return {"status": "duplicate", "event_id": ev["event_id"]}

        key = ev.get("location") or "_none_"
        hist = self._history[key]
        if hist and pd.Timestamp(ev["timestamp"]) < pd.Timestamp(hist[-1]["timestamp"]):
            ev["warnings"].append("Event arrived out of order; rolling features use the latest known history, "
                                  "which may be newer than this event.")

        anomaly = risk = None
        try:
            feats = self._features(ev)
            anomaly = self.scorer.score_anomaly(feats)
            risk = self.scorer.score_risk(feats)
        except ModelNotAvailable as exc:
            ev["warnings"].append(f"Anomaly scoring skipped: {exc}")
        except Exception as exc:  # noqa: BLE001
            self.db.log_error(ev["event_id"], f"scoring failed: {exc}", raw)
            ev["warnings"].append(f"Scoring failed: {exc}")

        alerts = alert_engine.generate_alerts(ev, anomaly, risk)
        self.db.insert_event(ev)
        if anomaly:
            self.db.insert_anomaly(ev["event_id"], anomaly["score"], anomaly["is_anomaly"],
                                   anomaly["severity"], anomaly["top_features"], anomaly["model_version"])
        if risk:
            self.db.insert_risk(ev["event_id"], risk["probability"], risk["level"],
                                risk["model_version"], risk["validated"])
        for a in alerts:
            self.db.insert_alert(a)
        hist.append({**ev})
        return {"status": "processed", "event_id": ev["event_id"], "category": alert_engine.classify_event(alerts),
                "alerts": len(alerts), "anomaly_score": anomaly["score"] if anomaly else None,
                "warnings": ev["warnings"], "source": ev["source"]}

    def process_many(self, events: Iterable[dict], delay: float = 0.0,
                     on_progress: Optional[Callable[[int, dict], None]] = None,
                     sort_by_time: bool = False) -> dict:
        evs = list(events)
        if sort_by_time:
            evs.sort(key=lambda e: str(e.get("timestamp")))
        summary = {"processed": 0, "duplicate": 0, "rejected": 0, "alerts": 0}
        for i, e in enumerate(evs, 1):
            res = self.process_event(e)
            summary[res["status"]] += 1
            summary["alerts"] += res.get("alerts", 0)
            if on_progress:
                on_progress(i, res)
            if delay > 0:
                time.sleep(delay)
        return summary


def replay_dataframe(df: pd.DataFrame, processor: EventProcessor, source: str = "replay",
                     limit: Optional[int] = None, delay: float = 0.0,
                     on_progress: Optional[Callable[[int, dict], None]] = None) -> dict:
    """Replay historical rows in time order to simulate a stream.

    ``source`` is stored on every event: use "replay" for real historical files
    and "synthetic" for generated demo data.
    """
    data = df.copy()
    if "timestamp" not in data.columns:
        raise ValidationError("Replay data needs a 'timestamp' column (see src/schema_mapping.py).")
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = data.sort_values("timestamp")
    if limit:
        data = data.head(limit)
    if "event_id" not in data.columns:
        data["event_id"] = [f"{source}-{i}" for i in range(len(data))]
    if "source" not in data.columns:
        data["source"] = source
    records = [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in data.to_dict("records")]
    return processor.process_many(records, delay=delay, on_progress=on_progress)


def main(argv: Optional[list] = None) -> int:
    import argparse
    from src import demo_data
    parser = argparse.ArgumentParser(description="Replay events through the pipeline.")
    parser.add_argument("--demo", action="store_true", help="Replay SYNTHETIC demo events")
    parser.add_argument("--file", help="Cleaned CSV with canonical columns to replay")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    proc = EventProcessor()
    if args.demo:
        df = demo_data.generate_demo_events().drop(columns=["injected_anomaly"])
        summary = replay_dataframe(df, proc, source="synthetic", limit=args.limit)
    elif args.file:
        from src.schema_mapping import to_canonical
        canon, mapping = to_canonical(pd.read_csv(args.file))
        print("Column mapping used:", mapping)
        canon["event_id"] = [f"replay-{i}" for i in range(len(canon))]
        summary = replay_dataframe(canon, proc, source="replay", limit=args.limit)
    else:
        print("Nothing to replay. Use --demo (synthetic) or --file <cleaned csv>.")
        return 0
    print("Summary:", summary)
    print("Database counts:", proc.db.counts())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
