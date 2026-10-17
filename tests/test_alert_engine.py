from src.alert_engine import anomaly_severity, generate_alerts, risk_level

PCTS = {"p1": -0.05, "p5": 0.0, "p50": 0.15}
EVENT = {"event_id": "e", "timestamp": "2024-01-01T00:00:00", "location": "A", "temperature": 30.0, "source": "replay"}


def test_anomaly_severity_levels():
    assert anomaly_severity(-0.1, PCTS) == "high"
    assert anomaly_severity(-0.01, PCTS) == "medium"
    assert anomaly_severity(0.1, PCTS) == "none"


def test_risk_levels():
    assert risk_level(0.2) == "low" and risk_level(0.5) == "elevated" and risk_level(0.9) == "high"


def test_normal_event_no_alerts():
    assert generate_alerts(EVENT, {"score": 0.1, "severity": "none"}, None) == []


def test_anomaly_alert_has_required_fields():
    a = generate_alerts(EVENT, {"score": -0.1, "severity": "high", "top_features": [{"feature": "temperature"}]}, None)[0]
    for k in ("alert_id", "timestamp", "location", "category", "severity", "evidence", "trigger_rule",
              "warnings", "recommendation"):
        assert k in a
    assert a["category"] == "environmental_anomaly"
    assert "pesticide" not in a["recommendation"].lower()


def test_anomaly_never_creates_pest_risk_alert():
    cats = {a["category"] for a in generate_alerts(EVENT, {"score": -1, "severity": "high"}, None)}
    assert cats == {"environmental_anomaly"}


def test_unvalidated_risk_ignored():
    risk = {"probability": 0.95, "level": "high", "validated": False}
    assert generate_alerts(EVENT, None, risk) == []


def test_validated_risk_alert():
    risk = {"probability": 0.95, "level": "high", "validated": True}
    assert generate_alerts(EVENT, None, risk)[0]["category"] == "high_pest_risk"


def test_confirmed_only_for_recorded_real_observation():
    real = dict(EVENT, pest_observed=1)
    assert [a["category"] for a in generate_alerts(real, None, None)] == ["confirmed_pest_observation"]
    synth = dict(real, source="synthetic")
    assert generate_alerts(synth, None, None) == []


def test_alert_ids_are_stable():
    a1 = generate_alerts(EVENT, {"score": -1, "severity": "high"}, None)[0]["alert_id"]
    a2 = generate_alerts(EVENT, {"score": -1, "severity": "high"}, None)[0]["alert_id"]
    assert a1 == a2
