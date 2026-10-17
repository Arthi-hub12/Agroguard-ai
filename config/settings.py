"""Central configuration for AgroGuard AI.

Every path and threshold lives here so you can change behaviour without
editing the rest of the code. Most values can also be overridden with
environment variables (useful on a server such as IBM LinuxONE).

Example (Windows PowerShell):
    $env:AGROGUARD_RAW_DIR = "D:\\my_datasets"

Example (Linux):
    export AGROGUARD_RAW_DIR=/home/linux1/datasets
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if not value:
        return default
    p = Path(value).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


# ---------------------------------------------------------------- paths
RAW_DIR = _env_path("AGROGUARD_RAW_DIR", PROJECT_ROOT / "data" / "raw")
CLEANED_DIR = _env_path("AGROGUARD_CLEANED_DIR", PROJECT_ROOT / "data" / "cleaned")
PROCESSED_DIR = _env_path("AGROGUARD_PROCESSED_DIR", PROJECT_ROOT / "data" / "processed")
MODELS_DIR = _env_path("AGROGUARD_MODELS_DIR", PROJECT_ROOT / "models")
REPORTS_DIR = _env_path("AGROGUARD_REPORTS_DIR", PROJECT_ROOT / "reports")
DB_PATH = _env_path("AGROGUARD_DB_PATH", PROJECT_ROOT / "data" / "processed" / "agroguard.db")

SUPPORTED_EXTENSIONS = {
    ".csv": "csv",
    ".tsv": "csv",
    ".xlsx": "excel",
    ".xls": "excel",
    ".json": "json",
    ".parquet": "parquet",
}

RANDOM_SEED = 42


# ------------------------------------------------------- cleaning rules
@dataclass
class CleaningConfig:
    """Reviewable cleaning decisions. Each one is reported in the cleaning report."""

    drop_fully_empty_rows: bool = True
    drop_fully_empty_columns: bool = True
    # Exact duplicates are only dropped if the dataset has NO key columns
    # that could make repeated rows meaningful (e.g. repeated daily readings).
    drop_exact_duplicates: bool = True
    # If a date column exists, identical rows on the same date are treated as
    # true duplicates; otherwise we keep them and only warn.
    keep_duplicates_when_no_date: bool = True
    # Columns with more missing values than this are flagged (never auto-dropped).
    high_missing_threshold: float = 0.5
    # Numeric missing values: "none" (leave NaN) or "median_within_location".
    numeric_imputation: str = "none"
    # Outliers are FLAGGED, never replaced, so real extreme weather is kept.
    outlier_iqr_multiplier: float = 3.0
    # Plausible physical ranges (inclusive). Values outside are flagged invalid.
    valid_ranges: dict = field(default_factory=lambda: {
        "temperature": (-60.0, 60.0),
        "humidity": (0.0, 100.0),
        "rainfall": (0.0, 2000.0),
        "wind_speed": (0.0, 120.0),
        "pressure": (800.0, 1100.0),
    })
    # Keywords used to guess what a column means.
    date_keywords: tuple = ("date", "time", "timestamp", "day", "datetime", "observed")
    location_keywords: tuple = ("location", "district", "state", "region", "city",
                                "village", "station", "site", "field", "county", "area")
    target_keywords: tuple = ("pest", "outbreak", "infest", "infestation", "damage",
                              "severity", "risk", "label", "target", "class")
    crop_keywords: tuple = ("crop", "plant", "variety", "commodity")


# --------------------------------------------------------- alert rules
@dataclass
class AlertConfig:
    """Thresholds for the alert engine. Anomaly severity and pest-risk
    severity are kept separate on purpose."""

    # Isolation Forest decision scores: LOWER means MORE unusual.
    # severity is decided by percentile of the training score distribution.
    anomaly_medium_percentile: float = 5.0   # score below the 5th percentile -> medium
    anomaly_high_percentile: float = 1.0     # score below the 1st percentile -> high
    risk_elevated_probability: float = 0.40
    risk_high_probability: float = 0.70


CLEANING = CleaningConfig()
ALERTS = AlertConfig()

# Schema for incoming events (field name -> required?)
EVENT_REQUIRED_FIELDS = ("event_id", "timestamp")
EVENT_OPTIONAL_FIELDS = ("location", "crop", "temperature", "humidity",
                         "rainfall", "wind_speed", "pressure", "pest_observed",
                         "pest_count", "source")
EVENT_WEATHER_FIELDS = ("temperature", "humidity", "rainfall", "wind_speed", "pressure")


def ensure_dirs() -> None:
    """Create output directories if they do not exist (never touches raw files)."""
    for d in (RAW_DIR, CLEANED_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
