"""Safe dataset integration.

Joins are only attempted when keys are detected in BOTH datasets. Before
merging we check that the right-hand table is unique on the join keys, so a
many-to-many join can never silently multiply rows. Join statistics are
always returned.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


class JoinError(ValueError):
    """Raised when a join would be unsafe."""


def find_common_keys(left: pd.DataFrame, right: pd.DataFrame, candidates: Optional[list] = None) -> list:
    cols = set(left.columns) & set(right.columns)
    if candidates is not None:
        return [c for c in candidates if c in cols]
    return sorted(cols)


def check_key_uniqueness(df: pd.DataFrame, keys: list) -> dict:
    if not keys:
        return {"unique": False, "duplicate_key_rows": 0, "reason": "no keys"}
    dup = int(df.duplicated(subset=keys, keep=False).sum())
    return {"unique": dup == 0, "duplicate_key_rows": dup}


def safe_left_join(left: pd.DataFrame, right: pd.DataFrame, keys: list,
                   suffixes: tuple = ("", "_right")) -> tuple[pd.DataFrame, dict]:
    """Left join that refuses to multiply rows.

    If the right table is not unique on ``keys`` the join is NOT performed
    (raises JoinError) - the caller must aggregate the right table first
    (see ``aggregate_to_keys``) or keep the datasets separate.
    """
    if not keys:
        raise JoinError("No join keys supplied. Datasets are kept separate.")
    missing_left = [k for k in keys if k not in left.columns]
    missing_right = [k for k in keys if k not in right.columns]
    if missing_left or missing_right:
        raise JoinError(f"Join keys missing - left: {missing_left}, right: {missing_right}")

    uniq = check_key_uniqueness(right, keys)
    if not uniq["unique"]:
        raise JoinError(f"Right table has {uniq['duplicate_key_rows']} rows with duplicate keys {keys}; "
                        "joining would multiply records. Aggregate it first.")

    rows_before = len(left)
    merged = left.merge(right, how="left", on=keys, suffixes=suffixes, indicator="_merge_status")
    if len(merged) != rows_before:  # defensive; should be impossible after the check above
        raise JoinError(f"Row count changed from {rows_before} to {len(merged)} during join.")

    matched = int((merged["_merge_status"] == "both").sum())
    merged = merged.drop(columns="_merge_status")
    right_keys = right[keys].drop_duplicates()
    left_keys = left[keys].drop_duplicates()
    unmatched_right = int(len(right_keys.merge(left_keys, on=keys, how="left", indicator=True)
                              .query("_merge == 'left_only'")))
    stats = {
        "keys": list(keys),
        "left_rows": rows_before,
        "right_rows": int(len(right)),
        "merged_rows": int(len(merged)),
        "matched_left_rows": matched,
        "unmatched_left_rows": rows_before - matched,
        "match_rate": round(matched / rows_before, 4) if rows_before else 0.0,
        "unmatched_right_keys": unmatched_right,
    }
    return merged, stats


def aggregate_to_keys(df: pd.DataFrame, keys: list, numeric_agg: str = "mean") -> pd.DataFrame:
    """Collapse a table to one row per key combination (numeric mean by default)."""
    num_cols = [c for c in df.select_dtypes("number").columns if c not in keys]
    return df.groupby(keys, as_index=False)[num_cols].agg(numeric_agg)


def try_integrate(pest: pd.DataFrame, weather: pd.DataFrame, keys: list) -> tuple[Optional[pd.DataFrame], dict]:
    """Attempt pest + weather integration; returns (merged or None, explanation)."""
    info: dict = {"attempted_keys": list(keys)}
    if not keys:
        info["result"] = "not_merged"
        info["reason"] = "No shared key columns (date/location). Datasets remain separate."
        return None, info
    try:
        weather_u = weather if check_key_uniqueness(weather, keys)["unique"] else aggregate_to_keys(weather, keys)
        if weather_u is not weather:
            info["note"] = "Weather table aggregated to one row per key before joining."
        merged, stats = safe_left_join(pest, weather_u, keys)
    except JoinError as exc:
        info["result"] = "not_merged"
        info["reason"] = str(exc)
        return None, info
    info["result"] = "merged"
    info["stats"] = stats
    if stats["match_rate"] < 0.5:
        info["warning"] = (f"Only {stats['match_rate']:.0%} of pest rows matched weather rows; "
                           "the merged dataset may be misleading. Check key formats.")
    return merged, info
