import numpy as np
import pandas as pd

from src.demo_data import generate_demo_events
from src.train_classifier import binarize_target, train_risk_model


def _real_frame(n=400, positives=60, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=n),
        "location": rng.choice(["A", "B", "C"], n),
        "temperature": rng.normal(27, 3, n), "humidity": rng.normal(65, 8, n),
        "rainfall": rng.gamma(1, 3, n), "source": "real"})
    y = np.zeros(n)
    y[rng.choice(n, positives, replace=False)] = 1
    df["pest_present"] = y
    return df


def test_refuses_synthetic_data(tmp_path):
    df = generate_demo_events(days=100)
    df["pest_present"] = 0
    r = train_risk_model(df, "pest_present", models_dir=tmp_path)
    assert r["supported"] is False and "synthetic" in r["reason"].lower()


def test_refuses_missing_target(tmp_path):
    r = train_risk_model(_real_frame(), "nope", models_dir=tmp_path)
    assert r["supported"] is False and r["data_needed"]


def test_refuses_single_class(tmp_path):
    df = _real_frame()
    df["pest_present"] = 0.0
    assert train_risk_model(df, "pest_present", models_dir=tmp_path)["supported"] is False


def test_refuses_too_few_rows(tmp_path):
    assert train_risk_model(_real_frame(n=50, positives=10), "pest_present", models_dir=tmp_path)["supported"] is False


def test_refuses_unclear_numeric_target_without_rule(tmp_path):
    df = _real_frame()
    df["pest_present"] = np.random.default_rng(1).integers(0, 50, len(df))
    r = train_risk_model(df, "pest_present", models_dir=tmp_path)
    assert r["supported"] is False and "threshold" in r["reason"].lower()


def test_binarize_text_labels():
    y, _ = binarize_target(pd.Series(["Yes", "no", "YES", "No"] * 5))
    assert y.tolist()[:4] == [1.0, 0.0, 1.0, 0.0]


def test_refuses_label_that_is_a_weather_rule(tmp_path):
    df = _real_frame(n=600)
    df["pest_present"] = ((df["humidity"] > 66) & (df["rainfall"] > 2.5)).astype(float)
    assert 20 < df["pest_present"].sum() < 580
    r = train_risk_model(df, "pest_present", models_dir=tmp_path)
    assert r["supported"] is False and "same-day weather" in r["reason"]
    assert not (tmp_path / "risk_model.joblib").exists()


def test_trains_with_valid_labels_and_never_uses_target_as_feature(tmp_path):
    df = _real_frame()
    r = train_risk_model(df, "pest_present", models_dir=tmp_path)
    assert r["supported"] is True
    assert "pest_present" not in r["features"] and "timestamp" not in r["features"]
    assert (tmp_path / "risk_model.joblib").exists()
