"""One command for the whole batch pipeline:

    python -m src.run_pipeline

discover + quality report -> clean -> train anomaly model -> replay events -> evaluation report.
Raw files are only read. Stops with a clear message if there is nothing to process.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from config import settings


def main(argv: Optional[list] = None) -> int:
    from src import cleaning, data_quality, evaluate
    from src.data_discovery import discover_datasets, print_discovery_summary
    from src.database import Database
    from src.event_processor import EventProcessor, replay_dataframe
    from src.inference import Scorer
    from src.schema_mapping import to_canonical
    from src.train_anomaly import InsufficientDataError, load_training_frame, train_anomaly_model

    settings.ensure_dirs()
    print("\n[1/5] Discovering datasets and writing the quality report")
    discovery = discover_datasets()
    print_discovery_summary(discovery)
    if not discovery.found_any:
        print("Nothing to process. Add dataset files to the folder above and run this again.")
        return 1
    data_quality.save_inventory(data_quality.build_inventory(discovery=discovery))

    print("\n[2/5] Cleaning (raw files are never changed)")
    summary = cleaning.clean_all()
    cleaned = [r["cleaned_path"] for r in summary["reports"] if r.get("cleaned_path")]
    if not cleaned:
        print("No dataset could be cleaned. See reports/cleaning_report.md.")
        return 1

    print("\n[3/5] Training the anomaly model")
    try:
        df, prov = load_training_frame(allow_synthetic=False)
        meta = train_anomaly_model(df, prov)
    except InsufficientDataError as exc:
        print(f"Cannot train: {exc}\nNeed a date column and at least two weather columns "
              "(temperature, humidity, rainfall, wind speed, pressure).")
        return 1
    Database().record_model_run("anomaly", meta)
    print(f"Trained {meta['model_version']} on {meta['n_training_rows']} rows ({meta['data_origin']})")

    print("\n[4/5] Replaying historical records as events")
    proc = EventProcessor(db=Database(), scorer=Scorer())
    for path in cleaned:
        canon, mapping = to_canonical(pd.read_csv(path))
        weather = [c for c in settings.EVENT_WEATHER_FIELDS if c in canon.columns]
        if "timestamp" not in canon.columns or len(weather) < 2:
            print(f"  skipped {Path(path).name}: no date column or fewer than 2 weather columns")
            continue
        stem = Path(path).stem
        canon["event_id"] = [f"{stem}-{i}" for i in range(len(canon))]
        print(f"  {Path(path).name}: column mapping {mapping}")
        print("  ", replay_dataframe(canon, proc, source="replay"))

    print("\n[5/5] Writing the evaluation report")
    print("  ", evaluate.write_report())
    print("\nDatabase counts:", proc.db.counts())
    print("Done. Start the dashboard with:  streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
