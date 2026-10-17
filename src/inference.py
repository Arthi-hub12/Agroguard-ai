"""Load saved models and score single events. Fails gracefully if models are missing."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from config import settings
from src import alert_engine


class ModelNotAvailable(RuntimeError):
    pass


class Scorer:
    """Wraps the anomaly model (and risk model, if one was validly trained)."""

    def __init__(self, models_dir: Optional[Path | str] = None) -> None:
        self.dir = Path(models_dir) if models_dir else Path(settings.MODELS_DIR)
        self.anomaly_pipe = None
        self.anomaly_meta: dict = {}
        self.risk_pipe = None
        self.risk_meta: dict = {}
        self._load()

    def _load(self) -> None:
        a_model, a_meta = self.dir / "anomaly_model.joblib", self.dir / "anomaly_model_meta.json"
        if a_model.exists() and a_meta.exists():
            try:
                self.anomaly_pipe = joblib.load(a_model)
                self.anomaly_meta = json.loads(a_meta.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                self.anomaly_pipe = None
                self.anomaly_meta = {"load_error": f"{type(exc).__name__}: {exc}. Retrain with "
                                                  "'python -m src.train_anomaly' (likely a scikit-learn version mismatch)."}
        r_model, r_meta = self.dir / "risk_model.joblib", self.dir / "risk_model_meta.json"
        if r_model.exists() and r_meta.exists():
            try:
                self.risk_pipe = joblib.load(r_model)
                self.risk_meta = json.loads(r_meta.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                self.risk_pipe = None

    @property
    def anomaly_ready(self) -> bool:
        return self.anomaly_pipe is not None

    @property
    def risk_ready(self) -> bool:
        return self.risk_pipe is not None and self.risk_meta.get("supported", False)

    def require_anomaly(self) -> None:
        if not self.anomaly_ready:
            msg = self.anomaly_meta.get("load_error") or (
                f"No anomaly model found in {self.dir}. Run: python -m src.train_anomaly")
            raise ModelNotAvailable(msg)

    def score_anomaly(self, feature_row: dict) -> dict:
        """feature_row: dict with (at least some of) the model's feature names."""
        self.require_anomaly()
        feats = self.anomaly_meta["features"]
        X = pd.DataFrame([{f: feature_row.get(f, np.nan) for f in feats}], columns=feats).astype(float)
        score = float(self.anomaly_pipe.decision_function(X)[0])
        pcts = self.anomaly_meta["score_percentiles"]
        severity = alert_engine.anomaly_severity(score, pcts)
        return {"score": score, "is_anomaly": severity != "none", "severity": severity,
                "top_features": self._explain(X.iloc[0]), "model_version": self.anomaly_meta.get("model_version")}

    def _explain(self, row: pd.Series, top_n: int = 3) -> list:
        """Features furthest from their training median (robust z-score). Context only."""
        ref = self.anomaly_meta.get("feature_reference", {})
        out = []
        for f, v in row.items():
            if pd.isna(v) or f not in ref:
                continue
            scale = (ref[f]["mad"] * 1.4826) or ref[f]["std"] or 1.0
            out.append({"feature": f, "value": float(v), "typical": ref[f]["median"],
                        "robust_z": round((float(v) - ref[f]["median"]) / scale, 2)})
        out.sort(key=lambda d: abs(d["robust_z"]), reverse=True)
        return out[:top_n]

    def score_risk(self, feature_row: dict) -> Optional[dict]:
        if not self.risk_ready:
            return None
        feats = self.risk_meta["features"]
        X = pd.DataFrame([{f: feature_row.get(f, np.nan) for f in feats}], columns=feats).astype(float)
        p = float(self.risk_pipe.predict_proba(X)[0, 1])
        return {"probability": p, "level": alert_engine.risk_level(p), "validated": True,
                "model_version": self.risk_meta.get("model_version")}


def self_test(models_dir: Optional[Path | str] = None) -> int:
    """Reload saved models and score a sample row (CLI: python -m src.inference)."""
    s = Scorer(models_dir)
    if not s.anomaly_ready:
        print("Anomaly model NOT available:", s.anomaly_meta.get("load_error") or "file missing. Run: python -m src.train_anomaly")
        return 1
    sample = {"temperature": 27.0, "humidity": 65.0, "rainfall": 0.0, "wind_speed": 8.0, "pressure": 1008.0}
    print("Anomaly model version:", s.anomaly_meta["model_version"], "| data origin:", s.anomaly_meta["data_origin"])
    print("Typical sample ->", s.score_anomaly(sample)["severity"])
    extreme = {"temperature": 45.0, "humidity": 5.0, "rainfall": 150.0, "wind_speed": 8.0, "pressure": 1008.0}
    print("Extreme sample ->", s.score_anomaly(extreme)["severity"])
    print("Risk model available:", s.risk_ready)
    return 0


if __name__ == "__main__":
    raise SystemExit(self_test())
