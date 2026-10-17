"""Reusable, configurable and fully logged dataset cleaning.

Principles
- Raw files are never modified; cleaned copies go to a separate folder.
- Outliers are FLAGGED (new boolean columns), never replaced or deleted,
  so genuine extreme weather is preserved.
- Duplicates are only removed when it is safe (see CleaningConfig).
- Every transformation is logged with the number of rows affected.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import settings
from config.settings import CleaningConfig
from src.data_discovery import discover_datasets, load_dataset
from src.data_quality import (detect_crop_columns, detect_date_columns,
                              detect_location_columns, detect_target_columns)


class CleaningLog:
    """Collects one entry per transformation."""

    def __init__(self) -> None:
        self.entries: list = []

    def add(self, step: str, affected_rows: int, detail: str = "", material: bool = False) -> None:
        self.entries.append({"step": step, "affected_rows": int(affected_rows),
                             "detail": detail, "material": bool(material)})

    def to_list(self) -> list:
        return list(self.entries)


def standardize_column_name(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unnamed"


def standardize_columns(df: pd.DataFrame, log: CleaningLog) -> tuple[pd.DataFrame, dict]:
    """Rename columns to snake_case. Returns the mapping original -> new so the
    original schema is never lost."""
    mapping: dict = {}
    seen: dict = {}
    for col in df.columns:
        new = standardize_column_name(col)
        if new in seen:
            seen[new] += 1
            new = f"{new}_{seen[new]}"
        else:
            seen[new] = 0
        mapping[str(col)] = new
    changed = sum(1 for k, v in mapping.items() if k != v)
    out = df.copy()
    out.columns = [mapping[str(c)] for c in df.columns]
    log.add("standardize_column_names", 0, f"{changed} of {len(mapping)} column names changed; "
            "mapping saved in the report.")
    return out, mapping


def normalize_text(value):
    if pd.isna(value):
        return value
    s = re.sub(r"\s+", " ", str(value).strip())
    return s


def clean_dataframe(df: pd.DataFrame, cfg: CleaningConfig = settings.CLEANING,
                    name: str = "dataset") -> tuple[pd.DataFrame, dict]:
    """Clean one DataFrame. Returns (cleaned_df, report_dict)."""
    log = CleaningLog()
    report: dict = {"dataset": name, "rows_in": int(len(df)), "columns_in": int(df.shape[1]),
                    "config": asdict(cfg)}

    if df.shape[1] == 0 or len(df) == 0:
        report.update({"rows_out": 0, "columns_out": int(df.shape[1]),
                       "log": [], "warnings": ["Dataset is empty; nothing to clean."],
                       "column_mapping": {}})
        return df.copy(), report

    work, mapping = standardize_columns(df, log)
    report["column_mapping"] = mapping

    # --- empty rows / columns
    if cfg.drop_fully_empty_rows:
        empty = work.isna().all(axis=1)
        if empty.any():
            work = work.loc[~empty].copy()
        log.add("drop_fully_empty_rows", int(empty.sum()))
    if cfg.drop_fully_empty_columns:
        empty_cols = [c for c in work.columns if work[c].isna().all()]
        if empty_cols:
            work = work.drop(columns=empty_cols)
        log.add("drop_fully_empty_columns", len(empty_cols), f"dropped: {empty_cols}", material=bool(empty_cols))

    # --- text normalisation
    text_cols = [c for c in work.columns if work[c].dtype == object or str(work[c].dtype).startswith("str")]
    for c in text_cols:
        work[c] = work[c].map(normalize_text)
    log.add("normalize_text", len(work) * len(text_cols), f"trimmed whitespace in {len(text_cols)} text columns")

    # categorical / location casing (Title Case for location, lower for crop) - consistent keys for joins
    loc_cols = [c for c in detect_location_columns(work, cfg) if c in text_cols]
    for c in loc_cols:
        work[c] = work[c].map(lambda v: v.title() if isinstance(v, str) else v)
    crop_cols = [c for c in detect_crop_columns(work, cfg) if c in text_cols]
    for c in crop_cols:
        work[c] = work[c].map(lambda v: v.lower() if isinstance(v, str) else v)
    if loc_cols or crop_cols:
        log.add("normalize_categories", 0, f"locations Title Case: {loc_cols}; crops lower case: {crop_cols}")

    # --- numeric conversion only where it is valid (>= 90% of non-null values convert)
    converted = []
    for c in text_cols:
        if c in loc_cols or c in crop_cols:
            continue
        non_null = work[c].dropna()
        if non_null.empty:
            continue
        num = pd.to_numeric(non_null.astype(str).str.replace(",", "", regex=False), errors="coerce")
        if num.notna().mean() >= 0.9:
            full = pd.to_numeric(work[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
            full[work[c].isna()] = np.nan
            lost = int((full.isna() & work[c].notna()).sum())
            work[c] = full
            converted.append(c)
            if lost:
                log.add("numeric_conversion_lost_values", lost, f"column '{c}': non-numeric entries became missing", material=True)
    log.add("convert_numeric_columns", 0, f"converted: {converted}")

    # --- dates
    date_cols = [c for c in detect_date_columns(work, cfg)]
    bad_dates_total = 0
    for c in date_cols:
        if pd.api.types.is_datetime64_any_dtype(work[c]):
            continue
        before_missing = int(work[c].isna().sum())
        parsed = pd.to_datetime(work[c], errors="coerce", format="mixed")
        bad = int(parsed.isna().sum()) - before_missing
        bad_dates_total += max(bad, 0)
        work[c] = parsed
        if bad > 0:
            log.add("date_parse_failures", bad, f"column '{c}': {bad} values could not be parsed and are now missing", material=True)
    log.add("parse_dates", 0, f"date columns: {date_cols}")

    # --- duplicates
    dup_mask = work.duplicated(keep="first")
    n_dup = int(dup_mask.sum())
    if n_dup:
        if cfg.drop_exact_duplicates and (date_cols or not cfg.keep_duplicates_when_no_date):
            work = work.loc[~dup_mask].copy()
            log.add("drop_exact_duplicates", n_dup,
                    "rows identical in EVERY column (dataset has a date column, so repeats are true duplicates)",
                    material=True)
        else:
            log.add("duplicates_kept", n_dup,
                    "exact duplicates found but kept because there is no date column to prove they are "
                    "accidental; review them manually", material=True)
    else:
        log.add("drop_exact_duplicates", 0, "no exact duplicates")

    # --- invalid ranges: flag, never delete
    flagged_cols = []
    # Column-name hints per range key, so "max_temp_c" is checked as a temperature.
    range_aliases = {
        "temperature": ("temperature", "temp"),
        "humidity": ("humidity",),
        "rainfall": ("rainfall", "rain", "precip"),
        "wind_speed": ("wind",),
        "pressure": ("pressure",),
    }
    for c in work.select_dtypes(include=[np.number]).columns:
        if c.endswith(("_invalid_range", "_outlier_flag")):
            continue
        for key, (lo, hi) in cfg.valid_ranges.items():
            if any(a in c for a in range_aliases.get(key, (key,))):
                bad = (work[c] < lo) | (work[c] > hi)
                n_bad = int(bad.sum())
                if n_bad:
                    work[f"{c}_invalid_range"] = bad
                    # values that are physically impossible become missing in the *value*
                    # column; the raw value stays in the raw file.
                    work.loc[bad, c] = np.nan
                    log.add("invalid_range_set_missing", n_bad,
                            f"column '{c}' outside [{lo}, {hi}]; set to missing and flagged in '{c}_invalid_range'",
                            material=True)
                    flagged_cols.append(c)
                break

    # --- outliers: flag only (IQR). Real extremes stay in the data.
    outlier_info = {}
    for c in work.select_dtypes(include=[np.number]).columns:
        if c.endswith("_invalid_range") or work[c].dropna().nunique() < 10:
            continue
        q1, q3 = work[c].quantile(0.25), work[c].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or pd.isna(iqr):
            continue
        lo, hi = q1 - cfg.outlier_iqr_multiplier * iqr, q3 + cfg.outlier_iqr_multiplier * iqr
        flag = (work[c] < lo) | (work[c] > hi)
        if flag.any():
            work[f"{c}_outlier_flag"] = flag
            outlier_info[c] = int(flag.sum())
    log.add("flag_outliers", sum(outlier_info.values()),
            f"flagged only (not changed or removed): {outlier_info}")

    # --- missing values: report; impute only if configured
    miss = work.isna().mean()
    high = {c: round(float(v), 3) for c, v in miss.items() if v >= cfg.high_missing_threshold}
    if cfg.numeric_imputation == "median_within_location" and loc_cols:
        n_filled = 0
        for c in work.select_dtypes(include=[np.number]).columns:
            if c.endswith(("_invalid_range", "_outlier_flag")):
                continue
            before = int(work[c].isna().sum())
            work[c] = work.groupby(loc_cols[0])[c].transform(lambda s: s.fillna(s.median()))
            n_filled += before - int(work[c].isna().sum())
        log.add("impute_numeric_median_within_location", n_filled,
                "median imputation inside each location; filled values are estimates", material=True)
    else:
        log.add("missing_values", int(work.isna().sum().sum()),
                "left as missing (no imputation) so models and reports can see the true gaps")

    report.update({
        "rows_out": int(len(work)),
        "columns_out": int(work.shape[1]),
        "log": log.to_list(),
        "high_missing_columns": high,
        "detected": {
            "date_columns": date_cols,
            "location_columns": loc_cols,
            "crop_columns": crop_cols,
            "target_columns_for_review": detect_target_columns(work, cfg),
        },
        "warnings": [],
    })
    if bad_dates_total:
        report["warnings"].append(f"{bad_dates_total} date values could not be parsed.")
    if high:
        report["warnings"].append(f"Columns with >= {cfg.high_missing_threshold:.0%} missing values: {list(high)}")
    return work, report


def clean_all(raw_dir: Optional[Path | str] = None, cleaned_dir: Optional[Path | str] = None,
              reports_dir: Optional[Path | str] = None,
              cfg: CleaningConfig = settings.CLEANING) -> dict:
    """Clean every discovered dataset independently. Raw files are only read."""
    cleaned_dir = Path(cleaned_dir) if cleaned_dir else Path(settings.CLEANED_DIR)
    reports_dir = Path(reports_dir) if reports_dir else Path(settings.REPORTS_DIR)
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    discovery = discover_datasets(raw_dir)
    reports = []
    for f in discovery.files:
        df, warns = load_dataset(f["path"])
        if df is None:
            reports.append({"dataset": f["name"], "skipped": True, "reason": warns})
            continue
        cleaned, rep = clean_dataframe(df, cfg, name=f["name"])
        rep["load_warnings"] = warns
        stem = Path(f["name"]).stem
        out_path = cleaned_dir / f"{stem}_cleaned.csv"
        # never write on top of a raw file
        if out_path.resolve() == Path(f["path"]).resolve():
            out_path = cleaned_dir / f"{stem}_cleaned_copy.csv"
        cleaned.to_csv(out_path, index=False)
        rep["cleaned_path"] = str(out_path)
        reports.append(rep)

    summary = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "datasets_found": len(discovery.files), "reports": reports,
               "discovery_messages": discovery.messages}
    (reports_dir / "cleaning_report.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (reports_dir / "cleaning_report.md").write_text(cleaning_report_markdown(summary), encoding="utf-8")
    return summary


def cleaning_report_markdown(summary: dict) -> str:
    lines = ["# Data cleaning report", "", f"Generated: {summary['generated_at']}", ""]
    if summary["datasets_found"] == 0:
        lines.append("**No datasets were found, so nothing was cleaned.**")
        lines += [f"- {m}" for m in summary.get("discovery_messages", [])]
        return "\n".join(lines) + "\n"
    for rep in summary["reports"]:
        lines += ["", f"## {rep['dataset']}", ""]
        if rep.get("skipped"):
            lines.append(f"Skipped: {rep['reason']}")
            continue
        lines += [f"Rows: {rep['rows_in']} -> {rep['rows_out']}; columns: {rep['columns_in']} -> {rep['columns_out']}", "",
                  "| Step | Rows affected | Detail | Material change |", "|---|---|---|---|"]
        for e in rep["log"]:
            lines.append(f"| {e['step']} | {e['affected_rows']} | {e['detail']} | {'YES' if e['material'] else ''} |")
        for w in rep.get("warnings", []):
            lines.append(f"\n- WARNING: {w}")
    return "\n".join(lines) + "\n"


def main(argv: Optional[list] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Clean all discovered datasets (raw files are never changed).")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--cleaned-dir", default=None)
    parser.add_argument("--reports-dir", default=None)
    args = parser.parse_args(argv)
    summary = clean_all(args.raw_dir, args.cleaned_dir, args.reports_dir)
    if summary["datasets_found"] == 0:
        print("No datasets found. See reports/cleaning_report.md and set AGROGUARD_RAW_DIR.")
    else:
        for r in summary["reports"]:
            print(r["dataset"], "->", r.get("cleaned_path", "skipped"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
