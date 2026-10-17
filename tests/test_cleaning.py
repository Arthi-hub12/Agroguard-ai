import hashlib

import numpy as np
import pandas as pd

from config.settings import CleaningConfig
from src.cleaning import clean_all, clean_dataframe, standardize_column_name


def test_column_names_standardised_with_mapping():
    df = pd.DataFrame({"Max Temp (C)": [1, 2], "District Name": ["a", "b"]})
    out, rep = clean_dataframe(df)
    assert "max_temp_c" in out.columns
    assert rep["column_mapping"]["Max Temp (C)"] == "max_temp_c"
    assert standardize_column_name("  A--B ") == "a_b"


def test_exact_duplicates_dropped_only_with_date():
    with_date = pd.DataFrame({"date": ["2024-01-01"] * 2, "temp": [20, 20]})
    out, _ = clean_dataframe(with_date)
    assert len(out) == 1
    no_date = pd.DataFrame({"x": [1, 1], "y": [2, 2]})
    out2, rep2 = clean_dataframe(no_date)
    assert len(out2) == 2  # kept: cannot prove they are accidental
    assert any(e["step"] == "duplicates_kept" for e in rep2["log"])


def test_repeated_readings_with_different_values_are_kept():
    df = pd.DataFrame({"date": ["2024-01-01"] * 2, "location": ["A", "A"], "temp": [20.0, 21.0]})
    out, _ = clean_dataframe(df)
    assert len(out) == 2


def test_invalid_range_flagged_but_extreme_valid_value_kept():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=30),
                       "temperature": [25.0] * 28 + [48.0, 999.0]})
    out, rep = clean_dataframe(df)
    assert out["temperature"].iloc[28] == 48.0           # legitimate heatwave kept
    assert np.isnan(out["temperature"].iloc[29])         # impossible value removed from value column
    assert out["temperature_invalid_range"].iloc[29]


def test_range_check_applies_to_aliased_column_names():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=5), "Max Temp (C)": [30.0, 31, -999, 29, 30]})
    out, rep = clean_dataframe(df)
    assert np.isnan(out["max_temp_c"].iloc[2]) and out["max_temp_c_invalid_range"].iloc[2]
    assert any(e["step"] == "invalid_range_set_missing" for e in rep["log"])


def test_outliers_flagged_not_changed():
    vals = list(np.linspace(10, 20, 40)) + [500.0]
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=41), "yield_value": vals})
    out, _ = clean_dataframe(df)
    assert out["yield_value"].iloc[-1] == 500.0
    assert out["yield_value_outlier_flag"].iloc[-1]


def test_bad_dates_become_missing_and_are_logged():
    df = pd.DataFrame({"date": ["2024-01-01", "not a date", "2024-01-03"], "v": [1, 2, 3]})
    out, rep = clean_dataframe(df)
    assert out["date"].isna().sum() == 1
    assert any(e["step"] == "date_parse_failures" for e in rep["log"])


def test_missing_values_not_blindly_dropped():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=4), "temperature": [20, np.nan, 22, np.nan]})
    out, _ = clean_dataframe(df, CleaningConfig())
    assert len(out) == 4 and out["temperature"].isna().sum() == 2


def test_empty_dataframe_safe():
    out, rep = clean_dataframe(pd.DataFrame())
    assert rep["rows_out"] == 0 and rep["warnings"]


def test_clean_all_never_modifies_raw(tmp_path):
    raw, cleaned, reports = tmp_path / "raw", tmp_path / "cleaned", tmp_path / "reports"
    raw.mkdir()
    f = raw / "w.csv"
    pd.DataFrame({"Date": ["2024-01-01", "2024-01-01"], "Temp": [20, 20]}).to_csv(f, index=False)
    before = hashlib.sha256(f.read_bytes()).hexdigest()
    res = clean_all(raw, cleaned, reports)
    assert hashlib.sha256(f.read_bytes()).hexdigest() == before
    assert (cleaned / "w_cleaned.csv").exists() and res["datasets_found"] == 1
    assert (reports / "cleaning_report.md").exists()


def test_clean_all_with_no_datasets(tmp_path):
    res = clean_all(tmp_path / "missing", tmp_path / "c", tmp_path / "r")
    assert res["datasets_found"] == 0
    assert "No datasets were found" in (tmp_path / "r" / "cleaning_report.md").read_text(encoding="utf-8")
