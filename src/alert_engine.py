"""Configurable alert rules.

Categories (kept separate on purpose):
  - normal
  - environmental_anomaly          (severity from anomaly score percentile)
  - elevated_pest_risk / high_pest_risk   (severity from risk probability; only when a validated model exists)
  - confirmed_pest_observation     (only when the event itself records pests)

Anomaly severity and pest-risk severity never mix. Recommendations only ask
for verification; no pesticide or treatment advice is given.
"""
from __future__ import annotations

import hashlib
from typing import Optional

from config import settings
from config.settings import AlertConfig

RECOMMENDATIONS = {
    "environmental_anomaly": "Check the weather station/data source for errors, then have the field inspected "
                             "if conditions look genuinely unusual. An anomaly does not mean pests are present.",
    "elevated_pest_risk": "Schedule a field inspection or scouting visit and consult a local agricultural expert to confirm.",
    "high_pest_risk": "Prioritise a field inspection and consult a local agricultural expert to confirm before any action.",
    "confirmed_pest_observation": "A pest was recorded in this observation. Verify with an agricultural expert and "
                                  "follow local extension guidance.",
}


def anomaly_severity(score: float, percentiles: dict, cfg: AlertConfig = settings.ALERTS) -> str:
    """Isolation Forest: lower score = more unusual. Compare with training percentiles."""
    if score <= percentiles.get("p1", float("-inf")):
        return "high"
    if score <= percentiles.get("p5", float("-inf")):
        return "medium"
    return "none"


def risk_level(probability: float, cfg: AlertConfig = settings.ALERTS) -> str:
    if probability >= cfg.risk_high_probability:
        return "high"
    if probability >= cfg.risk_elevated_probability:
        return "elevated"
    return "low"


def _alert_id(event_id: str, category: str) -> str:
    return hashlib.sha1(f"{event_id}|{category}".encode()).hexdigest()[:16]


def _base(event: dict, category: str, severity: str, rule: str, evidence: dict, warnings: list) -> dict:
    return {
        "alert_id": _alert_id(event["event_id"], category),
        "event_id": event["event_id"],
        "timestamp": event["timestamp"],
        "location": event.get("location"),
        "category": category,
        "severity": severity,
        "trigger_rule": rule,
        "evidence": evidence,
        "warnings": warnings,
        "recommendation": RECOMMENDATIONS[category],
        "source": event.get("source", "unknown"),
    }


def generate_alerts(event: dict, anomaly: Optional[dict], risk: Optional[dict],
                    cfg: AlertConfig = settings.ALERTS) -> list:
    """Return a list of alert dicts (possibly empty = normal)."""
    alerts = []
    warnings = list(event.get("warnings", []))
    weather = {k: event.get(k) for k in settings.EVENT_WEATHER_FIELDS if event.get(k) is not None}

    if anomaly and anomaly.get("severity") in ("medium", "high"):
        ev = {"observations": weather, "anomaly_score": anomaly["score"],
              "unusual_features": anomaly.get("top_features", [])}
        alerts.append(_base(event, "environmental_anomaly", anomaly["severity"],
                            f"Isolation Forest score {anomaly['score']:.3f} at/below training "
                            f"{'1st' if anomaly['severity'] == 'high' else '5th'} percentile",
                            ev, warnings))

    if risk and risk.get("validated") and risk.get("level") in ("elevated", "high"):
        cat = "high_pest_risk" if risk["level"] == "high" else "elevated_pest_risk"
        alerts.append(_base(event, cat, risk["level"],
                            f"Estimated risk score {risk['probability']:.2f} >= "
                            f"{cfg.risk_high_probability if risk['level'] == 'high' else cfg.risk_elevated_probability}",
                            {"observations": weather, "risk_score": risk["probability"],
                             "note": "Model score from historical patterns, not a confirmed infestation."}, warnings))

    if event.get("pest_observed") in (1, True) or (event.get("pest_count") or 0) > 0:
        if event.get("source") != "synthetic":
            alerts.append(_base(event, "confirmed_pest_observation", "medium",
                                "Event contains a recorded pest observation",
                                {"pest_observed": event.get("pest_observed"), "pest_count": event.get("pest_count")},
                                warnings))
    return alerts


def classify_event(alerts: list) -> str:
    return alerts[0]["category"] if alerts else "normal"
