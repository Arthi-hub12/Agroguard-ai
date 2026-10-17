"""Feature engineering without data leakage.

All rolling features use only PAST rows (``closed='left'`` / shift), so a
feature for time t never contains information from t or later.
Features are only created when the source columns exist.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from config import settings

WEATHER = settings.EVENT_WEATHER_FIELDS


def available_weather_columns(df: pd.DataFrame) -> list:
    return [c for c in WEATHER if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def add_calendar_features(df: pd.DataFrame, time_col: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    if time_col not in out.columns:
        return out
    ts = pd.to_datetime(out[time_col], errors="coerce")
    out["month"] = ts.dt.month
    out["day_of_year"] = ts.dt.dayofyear
    # cyclic encoding so December is close to January
    out["season_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 366.0)
    out["season_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 366.0)
    return out


def add_rolling_features(df: pd.DataFrame, time_col: str = "timestamp",
                         group_col: Optional[str] = "location",
                         windows: tuple = (3, 7)) -> pd.DataFrame:
    """Past-only rolling means / sums and 1-step changes, per location."""
    out = df.copy()
    if time_col not in out.columns:
        return out
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.sort_values([group_col, time_col] if group_col in out.columns else [time_col]).reset_index(drop=True)
    cols = available_weather_columns(out)
    groups = out.groupby(group_col, sort=False) if group_col in out.columns else None

    for c in cols:
        series_iter = (lambda w: groups[c].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())) \
            if groups is not None else (lambda w: out[c].shift(1).rolling(w, min_periods=1).mean())
        for w in windows:
            out[f"{c}_roll{w}_mean"] = series_iter(w)
        out[f"{c}_change"] = (groups[c].diff() if groups is not None else out[c].diff())
    if "rainfall" in cols:
        for w in windows:
            if groups is not None:
                out[f"rainfall_roll{w}_sum"] = groups["rainfall"].transform(
                    lambda s: s.shift(1).rolling(w, min_periods=1).sum())
            else:
                out[f"rainfall_roll{w}_sum"] = out["rainfall"].shift(1).rolling(w, min_periods=1).sum()
    return out


def add_pest_history(df: pd.DataFrame, time_col: str = "timestamp",
                     group_col: str = "location", pest_col: str = "pest_count") -> pd.DataFrame:
    """Previous observation of pests at the same location (past-only)."""
    out = df.copy()
    if pest_col not in out.columns or group_col not in out.columns or time_col not in out.columns:
        return out
    out = out.sort_values([group_col, time_col]).reset_index(drop=True)
    out[f"{pest_col}_prev"] = out.groupby(group_col)[pest_col].shift(1)
    return out


def build_feature_frame(df: pd.DataFrame, time_col: str = "timestamp",
                        group_col: str = "location") -> pd.DataFrame:
    out = add_calendar_features(df, time_col)
    out = add_rolling_features(out, time_col, group_col)
    out = add_pest_history(out, time_col, group_col)
    return out


def anomaly_feature_columns(df: pd.DataFrame, include_derived: bool = False) -> list:
    """Numeric environmental columns used by the anomaly model.

    By default only the raw weather measurements plus seasonal position are
    used. In a comparison on the SYNTHETIC demo data, adding the 19 rolling /
    change features dropped precision/recall from 0.88/1.00 to 0.41/0.47
    (the extra, mostly-imputed dimensions dilute the signal). Derived features
    are still built for the classifier. Re-check this choice on real data.
    """
    base = available_weather_columns(df)
    cal = [c for c in ("season_sin", "season_cos") if c in df.columns]
    extras = []
    if include_derived:
        extras = [c for c in df.columns
                  if c.endswith(("_roll3_mean", "_roll7_mean", "_change", "_roll3_sum", "_roll7_sum"))
                  and pd.api.types.is_numeric_dtype(df[c])]
    return base + cal + extras
