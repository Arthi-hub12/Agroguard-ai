import pytest

from src.database import Database


def _ev(i="e1"):
    return {"event_id": i, "timestamp": "2024-01-01T00:00:00", "location": "A", "temperature": 20.0,
            "source": "synthetic", "warnings": []}


def test_persistence_across_instances(tmp_path):
    p = tmp_path / "x.db"
    Database(p).insert_event(_ev())
    assert Database(p).counts()["events"] == 1  # reopened = restart


def test_duplicate_event_prevented(db):
    assert db.insert_event(_ev()) is True
    assert db.insert_event(_ev()) is False
    assert db.counts()["events"] == 1


def test_alert_insert_dedup_and_status(db):
    db.insert_event(_ev())
    alert = {"alert_id": "a1", "event_id": "e1", "timestamp": "2024-01-01", "category": "environmental_anomaly",
             "severity": "high"}
    assert db.insert_alert(alert) and not db.insert_alert(alert)
    assert db.update_alert_status("a1", "acknowledged")
    assert db.query_df("SELECT status FROM alerts")["status"].iloc[0] == "acknowledged"
    with pytest.raises(ValueError):
        db.update_alert_status("a1", "bogus")


def test_foreign_key_enforced(db):
    with pytest.raises(Exception):
        db.insert_alert({"alert_id": "a", "event_id": "missing", "timestamp": "t", "category": "c", "severity": "s"})


def test_backup_is_readable_copy(db, tmp_path):
    db.insert_event(_ev())
    b = db.backup(tmp_path / "bk")
    assert Database(b).counts()["events"] == 1


def test_sql_injection_like_text_is_safe(db):
    ev = _ev("e'; DROP TABLE events;--")
    db.insert_event(ev)
    assert db.counts()["events"] == 1
