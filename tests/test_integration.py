import pandas as pd
import pytest

from src.feature_engineering import add_rolling_features
from src.integration import JoinError, safe_left_join, try_integrate


def test_join_refuses_many_to_many():
    pest = pd.DataFrame({"date": ["d1", "d1"], "loc": ["A", "A"], "count": [1, 2]})
    weather = pd.DataFrame({"date": ["d1", "d1"], "loc": ["A", "A"], "temp": [20, 21]})
    with pytest.raises(JoinError):
        safe_left_join(pest, weather, ["date", "loc"])


def test_join_preserves_rows_and_reports_stats():
    pest = pd.DataFrame({"date": ["d1", "d2", "d3"], "loc": ["A", "A", "B"], "count": [1, 2, 3]})
    weather = pd.DataFrame({"date": ["d1", "d2", "d9"], "loc": ["A", "A", "A"], "temp": [20, 21, 22]})
    merged, stats = safe_left_join(pest, weather, ["date", "loc"])
    assert len(merged) == 3
    assert stats["matched_left_rows"] == 2 and stats["unmatched_left_rows"] == 1
    assert stats["unmatched_right_keys"] == 1


def test_try_integrate_no_keys_keeps_separate():
    merged, info = try_integrate(pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [1]}), [])
    assert merged is None and info["result"] == "not_merged"


def test_try_integrate_aggregates_duplicate_weather():
    pest = pd.DataFrame({"date": ["d1"], "count": [5]})
    weather = pd.DataFrame({"date": ["d1", "d1"], "temp": [20.0, 30.0]})
    merged, info = try_integrate(pest, weather, ["date"])
    assert len(merged) == 1 and merged["temp"].iloc[0] == 25.0 and "aggregated" in info["note"]


def test_rolling_features_do_not_leak_current_row():
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=4), "location": "A",
                       "temperature": [10.0, 20.0, 30.0, 1000.0]})
    out = add_rolling_features(df, windows=(3,))
    # feature for row 3 must only use rows 0..2, never the 1000 on that row
    assert out["temperature_roll3_mean"].iloc[3] == pytest.approx(20.0)
    assert pd.isna(out["temperature_roll3_mean"].iloc[0])
