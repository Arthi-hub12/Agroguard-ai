import pytest

from src.demo_data import generate_demo_events
from src.event_processor import EventProcessor, ValidationError, replay_dataframe, validate_event
from src.inference import ModelNotAvailable, Scorer
from src.train_anomaly import InsufficientDataError, load_training_frame, train_anomaly_model


@pytest.fixture()
def trained_scorer(tmp_path):
    df, prov = load_training_frame(tmp_path / "no_cleaned_here")
    assert prov["data_origin"] == "synthetic"
    train_anomaly_model(df, prov, tmp_path / "models")
    return Scorer(tmp_path / "models")


def test_validate_normalises(good_event):
    ev = validate_event(good_event)
    assert ev["location"] == "Field A" and ev["crop"] == "rice" and ev["source"] == "synthetic"


@pytest.mark.parametrize("bad", [
    {"timestamp": "2024-01-01", "temperature": 20},                       # no id
    {"event_id": "x", "temperature": 20},                                  # no timestamp
    {"event_id": "x", "timestamp": "garbage", "temperature": 20},          # invalid timestamp
    {"event_id": "x", "timestamp": "2024-01-01"},                          # no weather
    {"event_id": "x", "timestamp": "2024-01-01", "temperature": 999},      # only impossible value
])
def test_validate_rejects(bad):
    with pytest.raises(ValidationError):
        validate_event(bad)


def test_validate_missing_location_warns(good_event):
    good_event["location"] = None
    ev = validate_event(good_event)
    assert ev["location"] is None and any("Location missing" in w for w in ev["warnings"])


def test_missing_model_degrades_gracefully(db, tmp_path, good_event):
    proc = EventProcessor(db=db, scorer=Scorer(tmp_path / "empty"))
    res = proc.process_event(good_event)
    assert res["status"] == "processed" and any("skipped" in w for w in res["warnings"])
    with pytest.raises(ModelNotAvailable):
        Scorer(tmp_path / "empty").require_anomaly()


def test_duplicate_events_ignored(db, trained_scorer, good_event):
    proc = EventProcessor(db=db, scorer=trained_scorer)
    assert proc.process_event(good_event)["status"] == "processed"
    assert proc.process_event(good_event)["status"] == "duplicate"
    assert db.counts()["events"] == 1


def test_rejected_event_is_logged_not_raised(db, trained_scorer):
    proc = EventProcessor(db=db, scorer=trained_scorer)
    res = proc.process_event({"event_id": "bad", "timestamp": "nope", "temperature": 20})
    assert res["status"] == "rejected" and db.counts()["errors"] == 1


def test_extreme_event_creates_anomaly_alert_not_pest_alert(db, trained_scorer, good_event):
    proc = EventProcessor(db=db, scorer=trained_scorer)
    good_event.update(event_id="x1", temperature=48.0, humidity=8.0, rainfall=150.0)
    proc.process_event(good_event)
    cats = set(db.query_df("SELECT category FROM alerts")["category"])
    assert cats == {"environmental_anomaly"}


def test_out_of_order_event_warned(db, trained_scorer, good_event):
    proc = EventProcessor(db=db, scorer=trained_scorer)
    later = dict(good_event, event_id="late", timestamp="2024-06-10")
    earlier = dict(good_event, event_id="early", timestamp="2024-06-01")
    proc.process_event(later)
    res = proc.process_event(earlier)
    assert any("out of order" in w for w in res["warnings"])


def test_replay_and_restart_keeps_history(db, trained_scorer):
    df = generate_demo_events(days=40).drop(columns=["injected_anomaly"])
    proc = EventProcessor(db=db, scorer=trained_scorer)
    s = replay_dataframe(df, proc, source="synthetic")
    assert s["processed"] == 120
    proc2 = EventProcessor(db=db, scorer=trained_scorer)   # simulated restart
    assert db.counts()["events"] == 120
    assert replay_dataframe(df, proc2, source="synthetic")["duplicate"] == 120


def test_insufficient_training_rows(tmp_path):
    df = generate_demo_events(days=5)
    with pytest.raises(InsufficientDataError):
        train_anomaly_model(df.head(10), {"data_origin": "synthetic"}, tmp_path)


def test_no_real_data_and_synthetic_disallowed(tmp_path):
    with pytest.raises(InsufficientDataError):
        load_training_frame(tmp_path, allow_synthetic=False)


def test_corrupt_model_file_reports_retrain(tmp_path, good_event):
    (tmp_path / "anomaly_model.joblib").write_bytes(b"not a model")
    (tmp_path / "anomaly_model_meta.json").write_text("{}")
    s = Scorer(tmp_path)
    assert not s.anomaly_ready and "Retrain" in s.anomaly_meta["load_error"]
