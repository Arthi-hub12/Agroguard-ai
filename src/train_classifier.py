"""Pest-risk classifier (Random Forest) - trained ONLY when a defensible label exists.

A defensible label means: a column in REAL data recording actual pest
observations/outbreaks (binary or convertible to binary by an explicit rule
supplied by the user). We never derive a label from weather. If the checks
fail, the function returns a structured "not supported" result explaining what
data is needed.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from config import settings
from src.feature_engineering import build_feature_frame

MIN_ROWS = 200
MIN_POSITIVES = 20
# If a depth-4 decision tree on SAME-DAY weather alone reproduces the label this well
# (cross-validated PR-AUC), the label is almost certainly a weather rule, not pest data.
# Measured on one uploaded dataset: pest_risk definitions scored 0.998-1.000, while a noisy
# weather-dependent control label scored 0.21 (see README). 0.95 sits inside that gap.
RULE_LIKE_PR_AUC = 0.95


def label_reproducibility(data: pd.DataFrame, weather_cols: list, y_col: str = "_y",
                          seed: int = settings.RANDOM_SEED) -> Optional[float]:
    """Cross-validated PR-AUC of a tiny decision tree predicting the label from same-row weather.

    Returns None when it cannot be computed (no weather columns, too few cases).
    """
    if not weather_cols:
        return None
    y = data[y_col].astype(int)
    n_splits = min(5, int(y.sum()), int((1 - y).sum()))
    if n_splits < 2:
        return None
    X = SimpleImputer(strategy="median").fit_transform(data[weather_cols])
    proba = cross_val_predict(DecisionTreeClassifier(max_depth=4, random_state=seed), X, y,
                              cv=StratifiedKFold(n_splits, shuffle=True, random_state=seed),
                              method="predict_proba")[:, 1]
    return float(average_precision_score(y, proba))
MODEL_NAME = "risk_model.joblib"
META_NAME = "risk_model_meta.json"


def not_supported(reason: str, needs: Optional[list] = None) -> dict:
    return {"supported": False, "reason": reason,
            "data_needed": needs or [
                "Real historical pest observations with a date and location",
                "A clearly defined outbreak/infestation label (e.g. pest present yes/no, or count above a documented threshold)",
                "At least ~200 rows and ~20 positive cases across several locations and seasons",
            ]}


def binarize_target(series: pd.Series, positive_threshold: Optional[float] = None) -> tuple[Optional[pd.Series], str]:
    """Convert a target column to 0/1 using an EXPLICIT rule. Returns (y, description)."""
    s = series.copy()
    if s.dtype == object or str(s.dtype).startswith("str"):
        low = s.astype(str).str.strip().str.lower()
        truthy = {"1", "yes", "y", "true", "present", "infested", "outbreak", "high"}
        falsy = {"0", "no", "n", "false", "absent", "none", "low", "healthy"}
        known = low.isin(truthy | falsy)
        if known.mean() < 0.9:
            return None, "text labels are not clearly yes/no style; define a rule explicitly"
        y = low.where(known).map(lambda v: np.nan if pd.isna(v) else float(v in truthy))
        return y, "text labels mapped: yes/true/present/infested/outbreak/high -> 1, no/false/absent/none/low/healthy -> 0"
    num = pd.to_numeric(s, errors="coerce")
    uniq = set(num.dropna().unique().tolist())
    if uniq <= {0, 1, 0.0, 1.0}:
        return num, "numeric 0/1 label used as given"
    if positive_threshold is not None:
        return (num >= positive_threshold).astype(float).where(num.notna()), \
            f"user-defined rule: value >= {positive_threshold} -> 1"
    return None, "target is numeric but not 0/1; supply positive_threshold to define an outbreak explicitly"


def train_risk_model(df: pd.DataFrame, target_col: str, time_col: str = "timestamp",
                     group_col: str = "location", positive_threshold: Optional[float] = None,
                     models_dir: Optional[Path | str] = None, seed: int = settings.RANDOM_SEED) -> dict:
    models_dir = Path(models_dir) if models_dir else Path(settings.MODELS_DIR)

    if target_col not in df.columns:
        return not_supported(f"Target column '{target_col}' not found.")
    if df.get("source", pd.Series(dtype=object)).astype(str).eq("synthetic").any():
        return not_supported("Data contains synthetic rows; a validated risk model cannot be trained on synthetic data.")

    y_all, rule = binarize_target(df[target_col], positive_threshold)
    if y_all is None:
        return not_supported(f"Cannot define the label: {rule}.")
    data = df.copy()
    data["_y"] = y_all
    n_missing = int(data["_y"].isna().sum())
    data = data.dropna(subset=["_y"])
    if len(data) < MIN_ROWS:
        return not_supported(f"Only {len(data)} labelled rows (need >= {MIN_ROWS}).")
    pos = int(data["_y"].sum())
    if data["_y"].nunique() < 2:
        return not_supported("Only one class present in the target; a classifier cannot be trained.")
    if pos < MIN_POSITIVES or (len(data) - pos) < MIN_POSITIVES:
        return not_supported(f"Too few cases of one class (positives={pos}, negatives={len(data) - pos}).")

    data = build_feature_frame(data, time_col, group_col)
    # LEAKAGE GUARD: the target column itself and same-row pest counts are never features.
    drop = {"_y", target_col, time_col, "event_id", "source", "injected_anomaly", "pest_count", "pest_observed"}
    feats = [c for c in data.columns if c not in drop and pd.api.types.is_numeric_dtype(data[c])]
    if not feats:
        return not_supported("No numeric features available after leakage removal.")

    # DEFINITIONAL-LEAKAGE GUARD: is the "label" just a rule applied to the same day's weather?
    weather_cols = [c for c in settings.EVENT_WEATHER_FIELDS
                    if c in data.columns and pd.api.types.is_numeric_dtype(data[c])]
    repro = label_reproducibility(data, weather_cols)
    if repro is not None and repro >= RULE_LIKE_PR_AUC:
        return not_supported(
            f"The label '{target_col}' can be reproduced from same-day weather alone by a tiny decision tree "
            f"(cross-validated PR-AUC {repro:.3f} >= {RULE_LIKE_PR_AUC}). That usually means it was computed from "
            "weather by a rule instead of recorded from real pest observations. A model trained on it would only "
            "relearn that rule and its metrics would say nothing about real pest risk. If you know the label comes "
            "from field or trap records, check how it was produced; otherwise treat it as a rule-based index.",
            needs=["Pest observations recorded in the field (scouting, trap counts, farmer reports) with date and location",
                   "A documented outbreak definition that is NOT a formula over the same weather columns",
                   "At least ~200 rows and ~20 positive cases across several seasons and locations"])
    data = data.sort_values(time_col).reset_index(drop=True)
    n_dup = int(data.duplicated(subset=feats + ["_y"]).sum())

    # chronological holdout: last 25% by time is the test set
    cut = int(len(data) * 0.75)
    train, test = data.iloc[:cut], data.iloc[cut:]
    if train["_y"].nunique() < 2 or test["_y"].nunique() < 2:
        return not_supported("Chronological split leaves one class in train or test; need more varied history.")

    def make_pipe() -> Pipeline:
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("rf", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                                       random_state=seed, n_jobs=-1))])

    pipe = make_pipe().fit(train[feats], train["_y"])
    proba = pipe.predict_proba(test[feats])[:, 1]
    pred = (proba >= settings.ALERTS.risk_elevated_probability).astype(int)
    yt = test["_y"].astype(int).to_numpy()
    metrics = {
        "precision": float(precision_score(yt, pred, zero_division=0)),
        "recall": float(recall_score(yt, pred, zero_division=0)),
        "f1": float(f1_score(yt, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(yt, proba)),
        "baseline_positive_rate": float(yt.mean()),
        "confusion_matrix": confusion_matrix(yt, pred, labels=[0, 1]).tolist(),
        "threshold": settings.ALERTS.risk_elevated_probability,
    }

    grouped = None
    if group_col in data.columns and data[group_col].nunique() >= 3:
        gkf = GroupKFold(n_splits=min(3, data[group_col].nunique()))
        ps = []
        for tr, te in gkf.split(data[feats], data["_y"], data[group_col]):
            if data["_y"].iloc[tr].nunique() < 2 or data["_y"].iloc[te].nunique() < 2:
                continue
            p = make_pipe().fit(data[feats].iloc[tr], data["_y"].iloc[tr])
            ps.append(float(average_precision_score(data["_y"].iloc[te], p.predict_proba(data[feats].iloc[te])[:, 1])))
        grouped = {"pr_auc_by_location_fold": ps, "mean": float(np.mean(ps)) if ps else None}

    final = make_pipe().fit(data[feats], data["_y"])
    models_dir.mkdir(parents=True, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("risk-%Y%m%d%H%M%S")
    meta = {
        "supported": True, "model": "RandomForestClassifier", "model_version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target_column": target_col, "label_rule": rule, "features": feats,
        "n_rows": int(len(data)), "n_positive": pos, "n_missing_labels_dropped": n_missing,
        "duplicate_feature_rows": n_dup,
        "split": "chronological: first 75% train, last 25% test",
        "metrics_chronological_holdout": metrics, "grouped_by_location": grouped,
        "limitations": ["Metrics come from one historical holdout; not a guarantee of future performance.",
                        "Probabilities are model scores, not calibrated real-world probabilities."],
    }
    joblib.dump(final, models_dir / MODEL_NAME)
    (models_dir / META_NAME).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return meta


def main(argv: Optional[list] = None) -> int:
    import argparse
    from src.schema_mapping import to_canonical
    parser = argparse.ArgumentParser(description="Train the pest-risk classifier if valid labels exist.")
    parser.add_argument("--file", help="Cleaned CSV containing weather, location, date AND a pest label column")
    parser.add_argument("--target", help="Name of the label column")
    parser.add_argument("--positive-threshold", type=float, default=None)
    args = parser.parse_args(argv)
    if not args.file or not args.target:
        print(json.dumps(not_supported("No labelled file/target supplied. Use --file and --target."), indent=2))
        return 0
    raw = pd.read_csv(args.file)
    if args.target not in raw.columns:
        print(json.dumps(not_supported(f"Column '{args.target}' not in file."), indent=2))
        return 0
    canon, _ = to_canonical(raw)
    canon[args.target] = raw[args.target]
    result = train_risk_model(canon, args.target, positive_threshold=args.positive_threshold)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
