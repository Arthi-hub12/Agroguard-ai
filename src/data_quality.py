"""Data-quality inventory: what is in each dataset and how healthy is it."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import settings
from src.data_discovery import DiscoveryResult, discover_datasets, load_dataset


def _matches(name: str, keywords: tuple) -> bool:
    n = str(name).lower()
    return any(k in n for k in keywords)


def detect_date_columns(df: pd.DataFrame, cfg=settings.CLEANING) -> list:
    """Columns that look like dates, by name or by successful parsing."""
    found = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            found.append(str(col))
            continue
        if pd.api.types.is_numeric_dtype(series) and not _matches(col, cfg.date_keywords):
            continue
        if not (series.dtype == object or str(series.dtype).startswith("str")
                or pd.api.types.is_numeric_dtype(series)):
            continue
        by_name = _matches(col, cfg.date_keywords)
        sample = series.dropna().astype(str).head(200)
        if sample.empty:
            continue
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        ratio = float(parsed.notna().mean())
        # A column NAMED like a date is accepted at a lower parse rate (>= 50%) so that
        # unparseable values get reported by the cleaning step instead of being silently
        # left as text. Columns detected by content alone need >= 90%.
        if by_name and ratio >= 0.5:
            found.append(str(col))
        elif ratio >= 0.9 and not pd.api.types.is_numeric_dtype(series):
            found.append(str(col))
    return found


def detect_location_columns(df: pd.DataFrame, cfg=settings.CLEANING) -> list:
    return [str(c) for c in df.columns if _matches(c, cfg.location_keywords)]


def detect_target_columns(df: pd.DataFrame, cfg=settings.CLEANING) -> list:
    return [str(c) for c in df.columns if _matches(c, cfg.target_keywords)]


def detect_crop_columns(df: pd.DataFrame, cfg=settings.CLEANING) -> list:
    return [str(c) for c in df.columns if _matches(c, cfg.crop_keywords)]


def inventory_dataframe(df: pd.DataFrame, name: str = "dataset", path: str = "",
                        load_warnings: Optional[list] = None) -> dict:
    """Build the inventory dict for one DataFrame."""
    rows, cols = df.shape
    missing = df.isna().sum()
    columns = []
    for c in df.columns:
        columns.append({
            "name": str(c),
            "dtype": str(df[c].dtype),
            "missing": int(missing[c]),
            "missing_pct": round(float(missing[c]) / rows * 100, 2) if rows else 0.0,
            "unique": int(df[c].nunique(dropna=True)),
        })

    numeric = df.select_dtypes(include=[np.number])
    stats = {}
    if not numeric.empty and rows:
        desc = numeric.describe().T
        for col, rec in desc.iterrows():
            stats[str(col)] = {k: (None if pd.isna(v) else float(v)) for k, v in rec.items()}

    warnings = list(load_warnings or [])
    empty_rows = int(df.isna().all(axis=1).sum()) if rows else 0
    empty_cols = [str(c) for c in df.columns if df[c].isna().all()] if rows else []
    if rows == 0:
        warnings.append("Dataset has zero rows.")
    if empty_rows:
        warnings.append(f"{empty_rows} completely empty rows.")
    if empty_cols:
        warnings.append(f"Completely empty columns: {empty_cols}")
    dup_count = int(df.duplicated().sum()) if rows else 0
    if rows and dup_count / rows > 0.2:
        warnings.append(f"{dup_count} exact duplicate rows ({dup_count / rows:.0%}).")
    unnamed = [str(c) for c in df.columns if str(c).lower().startswith("unnamed")]
    if unnamed:
        warnings.append(f"Unnamed columns found (header may be misplaced): {unnamed}")
    for col in columns:
        if col["missing_pct"] >= settings.CLEANING.high_missing_threshold * 100:
            warnings.append(f"Column '{col['name']}' is {col['missing_pct']}% missing.")

    return {
        "name": name,
        "path": path,
        "rows": int(rows),
        "columns_count": int(cols),
        "columns": columns,
        "duplicate_rows": dup_count,
        "empty_rows": empty_rows,
        "empty_columns": empty_cols,
        "potential_date_columns": detect_date_columns(df),
        "potential_location_columns": detect_location_columns(df),
        "potential_crop_columns": detect_crop_columns(df),
        "potential_target_columns": detect_target_columns(df),
        "numeric_statistics": stats,
        "warnings": warnings,
    }


def build_inventory(raw_dir: Optional[Path | str] = None,
                    discovery: Optional[DiscoveryResult] = None) -> dict:
    """Inventory for every discovered dataset. Safe when there are none."""
    discovery = discovery or discover_datasets(raw_dir)
    datasets = []
    for f in discovery.files:
        df, warns = load_dataset(f["path"])
        if df is None:
            datasets.append({"name": f["name"], "path": f["path"], "rows": 0,
                             "columns_count": 0, "columns": [], "load_failed": True,
                             "warnings": warns})
            continue
        datasets.append(inventory_dataframe(df, f["name"], f["path"], warns))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "discovery": discovery.to_dict(),
        "dataset_count": len(datasets),
        "datasets": datasets,
    }


def inventory_to_markdown(inv: dict) -> str:
    lines = ["# Data inventory and quality report", "",
             f"Generated: {inv['generated_at']}", "",
             f"Searched folder: `{inv['discovery']['searched_dir']}`", "",
             f"Datasets found: **{inv['dataset_count']}**", ""]
    if inv["dataset_count"] == 0:
        lines.append("**No datasets were found.** No dataset contents are shown because none exist.")
        for m in inv["discovery"].get("messages", []):
            lines += ["", f"- {m}"]
        return "\n".join(lines) + "\n"
    for ds in inv["datasets"]:
        lines += ["", f"## {ds['name']}", "", f"Path: `{ds['path']}`", ""]
        if ds.get("load_failed"):
            lines.append("**Could not be loaded:**")
            lines += [f"- {w}" for w in ds["warnings"]]
            continue
        lines += [f"- Rows: {ds['rows']:,}", f"- Columns: {ds['columns_count']}",
                  f"- Exact duplicate rows: {ds['duplicate_rows']}",
                  f"- Date-like columns: {ds['potential_date_columns'] or 'none detected'}",
                  f"- Location-like columns: {ds['potential_location_columns'] or 'none detected'}",
                  f"- Crop-like columns: {ds['potential_crop_columns'] or 'none detected'}",
                  f"- Possible target columns (needs human review): "
                  f"{ds['potential_target_columns'] or 'none detected'}", "",
                  "| Column | Type | Missing | Missing % | Unique |", "|---|---|---|---|---|"]
        for c in ds["columns"]:
            lines.append(f"| {c['name']} | {c['dtype']} | {c['missing']} | {c['missing_pct']} | {c['unique']} |")
        if ds["warnings"]:
            lines += ["", "**Warnings**"] + [f"- {w}" for w in ds["warnings"]]
    return "\n".join(lines) + "\n"


def save_inventory(inv: dict, reports_dir: Optional[Path | str] = None) -> dict:
    """Write JSON and Markdown reports. Returns the written paths."""
    out = Path(reports_dir) if reports_dir else Path(settings.REPORTS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "data_inventory.json"
    md_path = out / "data_quality_report.md"
    json_path.write_text(json.dumps(inv, indent=2, default=str), encoding="utf-8")
    md_path.write_text(inventory_to_markdown(inv), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main(argv: Optional[list] = None) -> int:
    import argparse
    from src.data_discovery import print_discovery_summary

    parser = argparse.ArgumentParser(description="Discover datasets and write a quality report.")
    parser.add_argument("--raw-dir", default=None, help="Folder containing datasets (default from settings)")
    parser.add_argument("--reports-dir", default=None)
    args = parser.parse_args(argv)

    discovery = discover_datasets(args.raw_dir)
    print_discovery_summary(discovery)
    inv = build_inventory(discovery=discovery)
    paths = save_inventory(inv, args.reports_dir)
    print(f"Reports written: {paths}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
