"""Map arbitrary cleaned-dataset columns onto AgroGuard's canonical event schema.

Mapping is based on column-name keywords only and is always reported, so a
human can check it. Nothing is guessed for values: if a canonical field has no
matching column it is simply absent.
"""
from __future__ import annotations

import pandas as pd

ALIASES = {
    "timestamp": ("timestamp", "datetime", "date", "time", "observed"),
    "location": ("location", "district", "region", "state", "station", "site", "city", "village"),
    "crop": ("crop", "plant", "commodity"),
    "temperature": ("temperature", "temp", "tavg", "t2m"),
    "humidity": ("humidity", "rh"),
    "rainfall": ("rainfall", "rain", "precip", "precipitation"),
    "wind_speed": ("wind_speed", "windspeed", "wind"),
    "pressure": ("pressure",),
}
_EXCLUDE_SUFFIXES = ("_invalid_range", "_outlier_flag")


def suggest_mapping(df: pd.DataFrame) -> dict:
    """Return {canonical_field: source_column}. Prefers exact matches, then
    the first keyword hit; never maps two fields to the same column."""
    cols = [c for c in df.columns if not str(c).endswith(_EXCLUDE_SUFFIXES)]
    mapping: dict = {}
    used: set = set()
    for field, aliases in ALIASES.items():
        chosen = None
        for a in aliases:  # exact first
            if a in cols and a not in used:
                chosen = a
                break
        if chosen is None:
            for c in cols:
                if c in used:
                    continue
                low = str(c).lower()
                if any(a in low for a in aliases):
                    if field != "timestamp" and field not in ("location", "crop") and \
                            not pd.api.types.is_numeric_dtype(df[c]):
                        continue
                    chosen = c
                    break
        if chosen is not None:
            mapping[field] = chosen
            used.add(chosen)
    return mapping


def to_canonical(df: pd.DataFrame, mapping: dict | None = None) -> tuple[pd.DataFrame, dict]:
    mapping = mapping if mapping is not None else suggest_mapping(df)
    out = pd.DataFrame(index=df.index)
    for field, src in mapping.items():
        out[field] = df[src]
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    return out, mapping
