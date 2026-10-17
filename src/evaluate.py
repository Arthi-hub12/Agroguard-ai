"""Write a model evaluation report from the saved model metadata (no invented numbers)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import settings


def _load(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, ValueError):
        return None


def build_report(models_dir: Optional[Path | str] = None) -> str:
    d = Path(models_dir) if models_dir else Path(settings.MODELS_DIR)
    anomaly = _load(d / "anomaly_model_meta.json")
    risk = _load(d / "risk_model_meta.json")
    L = ["# Model evaluation report", "", f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}", ""]

    L += ["## Environmental anomaly model (Isolation Forest)", ""]
    if not anomaly:
        L.append("No anomaly model has been trained yet. Run `python -m src.train_anomaly`.")
    else:
        L += [f"- Version: `{anomaly['model_version']}`", f"- Trained at: {anomaly['trained_at']}",
              f"- Training data origin: **{anomaly['data_origin'].upper()}**",
              f"- Training rows: {anomaly['n_training_rows']}", f"- Features: {', '.join(anomaly['features'])}",
              f"- Parameters: {anomaly['parameters']}"]
        if anomaly["data_origin"] == "synthetic":
            L.append("- **WARNING: trained on synthetic demonstration data. Scores show how the pipeline works, "
                     "not real-world behaviour.**")
        if anomaly["data_origin"] == "user_provided":
            L.append("- Note: " + anomaly.get("provenance", {}).get(
                "note", "data was supplied by the user; authenticity not verified."))
        ev = anomaly.get("evaluation")
        if ev:
            n_ref = ev.get("n_reference_positives", ev.get("n_injected"))
            L += ["", f"Evaluation ({ev['kind']}):", f"- Precision: {ev['precision']:.2f}",
                  f"- Recall: {ev['recall']:.2f}", f"- Reference positives: {n_ref}"]
            if "pr_auc" in ev:
                L += [f"- PR-AUC: {ev['pr_auc']:.2f} (a random ranking would score about {ev['base_rate']:.2f})",
                      f"- Observations flagged by the model: {ev.get('n_flagged')}"]
            L.append("In-sample comparison; the reference labels were not used for training. Indicative only.")
        else:
            L.append("- No reviewed observations were available, so no accuracy metrics are reported.")
        L += ["", "An anomaly is an unusual observation. It is not evidence that pests are present."]

    L += ["", "## Pest-risk model (Random Forest)", ""]
    if not risk or not risk.get("supported"):
        L.append("**Not trained.** A supervised model requires real historical pest labels. "
                 "No pest-risk scores are produced and none are shown.")
        L += ["", "Data needed:", "- Real pest observations with date and location",
              "- A documented outbreak definition", "- ~200+ rows and ~20+ positive cases across seasons and locations"]
    else:
        m = risk["metrics_chronological_holdout"]
        L += [f"- Version: `{risk['model_version']}`", f"- Label rule: {risk['label_rule']}", f"- Split: {risk['split']}",
              f"- Rows: {risk['n_rows']} (positives: {risk['n_positive']})", "",
              f"| Precision | Recall | F1 | PR-AUC | Baseline positive rate |", "|---|---|---|---|---|",
              f"| {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {m['pr_auc']:.2f} | {m['baseline_positive_rate']:.2f} |",
              "", f"Confusion matrix [[TN, FP],[FN, TP]]: {m['confusion_matrix']}"]
        if risk.get("grouped_by_location"):
            L.append(f"Grouped-by-location PR-AUC: {risk['grouped_by_location']}")
        L += [""] + [f"- {x}" for x in risk.get("limitations", [])]
    return "\n".join(L) + "\n"


def write_report(models_dir: Optional[Path | str] = None, reports_dir: Optional[Path | str] = None) -> Path:
    out = Path(reports_dir) if reports_dir else Path(settings.REPORTS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "model_evaluation_report.md"
    p.write_text(build_report(models_dir), encoding="utf-8")
    return p


if __name__ == "__main__":
    print(f"Wrote {write_report()}")
