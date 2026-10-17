import numpy as np
import pandas as pd

from src.data_discovery import discover_datasets, load_dataset
from src.data_quality import build_inventory, inventory_dataframe, save_inventory


def test_discovery_missing_folder(tmp_path):
    r = discover_datasets(tmp_path / "nope")
    assert not r.exists and not r.found_any
    assert any("does not exist" in m for m in r.messages)


def test_discovery_empty_folder_gives_guidance(tmp_path):
    r = discover_datasets(tmp_path)
    assert r.exists and not r.found_any
    assert any("AGROGUARD_RAW_DIR" in m for m in r.messages)


def test_discovery_recursive_and_types(tmp_path):
    (tmp_path / "sub").mkdir()
    pd.DataFrame({"a": [1]}).to_csv(tmp_path / "sub" / "x.csv", index=False)
    (tmp_path / "ignore.txt").write_text("hi")
    r = discover_datasets(tmp_path)
    assert len(r.files) == 1 and r.counts_by_type == {"csv": 1}


def test_empty_file_handled(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("")
    df, warns = load_dataset(p)
    assert df is None and any("empty" in w.lower() for w in warns)


def test_malformed_json_does_not_crash(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    df, warns = load_dataset(p)
    assert df is None and warns


def test_inventory_missing_and_duplicates_and_dates():
    df = pd.DataFrame({"Date": ["2024-01-01", "2024-01-02", "2024-01-02"], "District": ["A", "B", "B"],
                       "Temp": [20.0, np.nan, np.nan]})
    inv = inventory_dataframe(df, "t")
    assert inv["duplicate_rows"] == 1
    assert "Date" in inv["potential_date_columns"]
    assert "District" in inv["potential_location_columns"]
    temp = next(c for c in inv["columns"] if c["name"] == "Temp")
    assert temp["missing"] == 2 and temp["missing_pct"] == 66.67


def test_inventory_with_no_datasets_is_safe(tmp_path):
    inv = build_inventory(tmp_path)
    assert inv["dataset_count"] == 0
    paths = save_inventory(inv, tmp_path / "reports")
    text = open(paths["markdown"], encoding="utf-8").read()
    assert "No datasets were found" in text
