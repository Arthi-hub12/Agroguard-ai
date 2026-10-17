"""Train the environmental anomaly detector (Isolation Forest).

An anomaly = an UNUSUAL observation compared with the training data. It is not
proof of a pest outbreak.

Training data priority:
  1. Cleaned real datasets that map to weather columns (data/cleaned/*.csv).
  2. If none exist, SYNTHETIC demo data (the saved model is labelled synthetic).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import settings
from src import demo_data
from src.feature_engineering import anomaly_feature_columns, build_feature_frame
from src.schema_mapping import to_canonical

MIN_TRAIN_ROWS = 50
MODEL_NAME = "anomaly_model.joblib"
META_NAME = "anomaly_model_meta.json"


class InsufficientDataError(ValueError):
    pass


def load_training_frame(cleaned_dir: Optional[Path | str] = None, allow_synthetic: bool = True) -> tuple[pd.DataFrame, dict]:
    """Find weather-like data. Returns (canonical frame, provenance dict)."""
    cleaned_dir = Path(cleaned_dir) if cleaned_dir else Path(settings.CLEANED_DIR)
    frames, used_files = [], []
    for p in sorted(cleaned_dir.glob("*_cleaned*.csv")):
        try:
            raw = pd.read_csv(p)
        except Exception:  # noqa: BLE001
            continue
        canon, mapping = to_canonical(raw)
        weather = [c for c in settings.EVENT_WEATHER_FIELDS if c in canon.columns]
        if "timestamp" in canon.columns and len(weather) >= 2 and canon["timestamp"].notna().any():
            canon["source"] = "user_provided"
            # Optional 0/1 reference column (not used for training, only for a caveated evaluation).
            if "anomaly_label" in raw.columns and set(raw["anomaly_label"].dropna().unique()) <= {0, 1}:
                canon["reference_anomaly"] = raw["anomaly_label"].fillna(0).astype(bool)
            frames.append(canon)
            used_files.append({"file": p.name, "mapping": mapping})
    if frames:
        return pd.concat(frames, ignore_index=True), {
            "data_origin": "user_provided", "files": used_files,
            "note": "Supplied by the user. Whether the measurements are real observations has not been verified."}
    if not allow_synthetic:
        raise InsufficientDataError("No cleaned real weather data found (need a date column and at least 2 weather columns).")
    df = demo_data.generate_demo_events()
    return df, {"data_origin": "synthetic", "files": [],
                "note": "No usable real weather data found; trained on SYNTHETIC demo data. "
                        "Scores demonstrate the pipeline only."}


def train_anomaly_model(df: pd.DataFrame, provenance: dict, models_dir: Optional[Path | str] = None,
                        contamination: float = 0.03, seed: int = settings.RANDOM_SEED) -> dict:
    models_dir = Path(models_dir) if models_dir else Path(settings.MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)

    feat_df = build_feature_frame(df)
    features = anomaly_feature_columns(feat_df)
    if not features:
        raise InsufficientDataError("No numeric environmental feature columns available.")
    X = feat_df[features].replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.notna().any()]
    features = list(X.columns)
    if len(X) < MIN_TRAIN_ROWS:
        raise InsufficientDataError(f"Only {len(X)} rows; need at least {MIN_TRAIN_ROWS} to train.")

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("forest", IsolationForest(n_estimators=200, contamination=contamination, random_state=seed)),
    ])
    pipe.fit(X)
    scores = pipe.decision_function(X)  # lower = more unusual
    pcts = {f"p{p}": float(np.percentile(scores, p)) for p in (1, 5, 50)}

    # per-feature reference statistics used to explain WHY something looked unusual
    ref = {c: {"median": float(X[c].median()), "mad": float((X[c] - X[c].median()).abs().median()),
               "std": float(X[c].std() or 1.0)} for c in features}

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "IsolationForest",
        "model_version": datetime.now(timezone.utc).strftime("anomaly-%Y%m%d%H%M%S"),
        "data_origin": provenance.get("data_origin"),
        "provenance": provenance,
        "n_training_rows": int(len(X)),
        "features": features,
        "parameters": {"n_estimators": 200, "contamination": contamination, "random_state": seed},
        "score_percentiles": pcts,
        "feature_reference": ref,
        "flagged_in_training": int((pipe.predict(X) == -1).sum()),
        "evaluation": None,
        "limitations": [
            "An anomaly is an unusual observation, NOT evidence of pests.",
            "Isolation Forest is unsupervised; there is no ground truth unless reviewed observations exist.",
        ],
    }

    # Evaluation only when injected (synthetic) or reference labels are present. The labels are
    # never used for training; this is an in-sample comparison, so treat it as indicative only.
    ref_col = next((c for c in ("injected_anomaly", "reference_anomaly") if c in feat_df.columns), None)
    if ref_col:
        from sklearn.metrics import average_precision_score, precision_score, recall_score
        y = feat_df[ref_col].astype(bool).to_numpy()
        if 0 < y.sum() < len(y):
            pred = pipe.predict(X) == -1
            if ref_col == "injected_anomaly":
                kind = "SYNTHETIC injected anomalies (demonstration only - not real-world accuracy)"
            else:
                kind = ("Agreement with the dataset's own 'anomaly_label' column. How that label was produced is "
                        "unknown (it may be a threshold rule on the same weather values), so this measures "
                        "agreement with that label, not real-world accuracy")
            meta["evaluation"] = {
                "kind": kind,
                "precision": float(precision_score(y, pred, zero_division=0)),
                "recall": float(recall_score(y, pred, zero_division=0)),
                "pr_auc": float(average_precision_score(y, -scores)),   # threshold-free; lower score = more unusual
                "base_rate": float(y.mean()),                          # PR-AUC of a random ranking would be ~this
                "n_reference_positives": int(y.sum()),
                "n_flagged": int(pred.sum()),
            }

    joblib.dump(pipe, models_dir / MODEL_NAME)
    (models_dir / META_NAME).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return meta


def main(argv: Optional[list] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Train the anomaly model.")
    parser.add_argument("--cleaned-dir", default=None)
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--no-synthetic", action="store_true", help="Fail instead of using synthetic demo data.")
    args = parser.parse_args(argv)
    try:
        df, prov = load_training_frame(args.cleaned_dir, allow_synthetic=not args.no_synthetic)
        meta = train_anomaly_model(df, prov, args.models_dir)
    except InsufficientDataError as exc:
        print(f"Cannot train anomaly model: {exc}")
        return 1
    print(f"Trained {meta['model_version']} on {meta['n_training_rows']} rows ({meta['data_origin']} data)")
    if meta["evaluation"]:
        print("Evaluation:", meta["evaluation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
