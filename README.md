# AgroGuard AI

A software-only early-warning prototype for agricultural **environmental anomalies** and (when real
labelled data exists) **estimated pest risk**. It cleans datasets, trains models, replays historical
records as events, raises alerts, stores everything in SQLite and shows it in a Streamlit dashboard.

> **Read this first - what is real and what is not**
>
> | Output | Meaning |
> |---|---|
> | **Environmental anomaly** | An *unusual observation* compared with the training data. **Not** evidence of pests. |
> | **Estimated pest risk** | A model score. Only produced if a model was trained on **real, labelled** pest data. Until then the dashboard says "not available". |
> | **Confirmed pest observation** | Only when an incoming real record itself contains a pest observation. |
> | **Synthetic / demo** | Generated data (`source = synthetic`). Shows how the system works. Metrics from it are **not** real-world accuracy. |
>
> "Events" come from **replaying files or generated demo data**. This is not a live sensor network or production
> streaming system.

## Current status (honest summary)

- **Dataset tested:** `AgroGuard_AI_Cleaned.csv` supplied by the project owner: 2,192 daily weather rows
  (2019-2024), one location, no gaps. It was run through the full pipeline (discovery, cleaning, anomaly
  model, replay of 2,192 events, 110 alerts, dashboard).
- **The `pest_risk` column is NOT used to train a model.** A depth-4 decision tree using only the same day's
  weather reproduces it with ~99.8-100% cross-validated PR-AUC (a deliberately noisy weather-dependent control
  label scored 0.21), so it looks computed from weather by a rule rather than recorded from pest observations.
  `src/train_classifier.py` refuses such labels. The dashboard shows the column only as a labelled, informational chart.
- **Anomaly model vs the dataset's `anomaly_label`:** precision 0.18, recall 0.38, PR-AUC 0.17 (random ~0.01).
  That is better than chance but not a close match, and the label's origin is unknown (it also looks
  threshold-like), so this is indicative only. The model was not tuned to that label.
- **Alert volume:** "medium" severity means below the 5th percentile of training scores, so roughly 1 day in 20 raises
  an alert (110 alerts over 2,192 days). Change the percentiles in `AlertConfig` if that is too many.
- Authenticity of the weather measurements is unverified (the data is labelled `user_provided`, not "real").
- Tested on Linux x86_64 / Python 3.13 (62 automated tests passing). **Not yet tested** on Windows, Python 3.11 or
  IBM LinuxONE (s390x). See `docs/DEPLOY_LINUXONE.md`.

## Project layout

```
app.py                  Streamlit dashboard (7 pages)
config/settings.py      Paths, cleaning rules, alert thresholds (env-var overridable)
src/data_discovery.py   Finds CSV/Excel/JSON/Parquet files recursively; safe loading
src/data_quality.py     Inventory + quality report (JSON and Markdown)
src/cleaning.py         Configurable, logged cleaning; raw files never modified
src/integration.py      Join checks (refuses many-to-many), join statistics
src/feature_engineering.py  Past-only rolling features (no leakage)
src/schema_mapping.py   Maps arbitrary columns to AgroGuard's event schema
src/run_pipeline.py     One command: quality -> clean -> train -> replay -> report
src/train_anomaly.py    Isolation Forest
src/train_classifier.py Random Forest - only with valid real labels
src/event_processor.py  Validate -> dedupe -> score -> alert -> store; replay mode
src/alert_engine.py     Rules and thresholds
src/database.py         SQLite schema, backup
src/inference.py        Load models, score events, self-test
src/evaluate.py         Evaluation report from saved metadata
src/demo_data.py        SYNTHETIC data generator
tests/                  pytest suite
docs/                   Deployment, troubleshooting, data sources, demo guide
```

## Setup (local)

```bash
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# Linux/macOS:         source .venv/bin/activate
pip install -r requirements.txt
```

## Quickest way

```bash
# put your dataset file(s) in data/raw/ first
python -m src.run_pipeline      # quality report, cleaning, anomaly model, replay, evaluation report
streamlit run app.py
```

## Run it, stage by stage

```bash
# 1. Put your datasets in data/raw/  (or: export AGROGUARD_RAW_DIR=/path ; PowerShell: $env:AGROGUARD_RAW_DIR="D:\data")
python -m src.data_quality          # discovery + reports/data_quality_report.md (works even if the folder is empty)

# 2. Clean (writes data/cleaned/, never touches data/raw/)
python -m src.cleaning              # reports/cleaning_report.md

# 3. Train the anomaly model (uses cleaned real weather data; falls back to SYNTHETIC if none, and says so)
python -m src.train_anomaly
python -m src.inference             # reload + test the saved model

# 4. Optional: supervised pest risk - ONLY with real labels
python -m src.train_classifier --file data/cleaned/your_file_cleaned.csv --target pest_present
#    (add --positive-threshold N if the label is a count and you define "outbreak" as count >= N)

# 5. Replay events through the pipeline
python -m src.event_processor --demo                      # SYNTHETIC
python -m src.event_processor --file data/cleaned/x_cleaned.csv --limit 500   # real historical replay

# 6. Evaluation report and dashboard
python -m src.evaluate
streamlit run app.py
```

Demo without any data: `python -m src.train_anomaly && python -m src.event_processor --demo && streamlit run app.py`

## Tests

```bash
python -m pytest -q
```

Covers discovery (empty/missing folders, empty/malformed files), cleaning (raw files untouched, outliers kept,
invalid ranges flagged, bad dates), joins (many-to-many refused), leakage, classifier refusal cases, event
validation, duplicates, alerts, database persistence and missing/corrupt model files.

## Design decisions you can change

- Cleaning rules: `config/settings.py` -> `CleaningConfig` (each material change is listed in the report).
- Alert thresholds: `AlertConfig`. Anomaly severity is based on training-score percentiles (1st = high, 5th = medium).
- The anomaly model uses raw weather + season only. In a test on synthetic data, adding 19 rolling/change features
  cut precision/recall from 0.88/1.00 to 0.41/0.47. **Re-check on real data.**

## Known limitations

- Anomaly metrics on synthetic data measure detection of artificial spikes only.
- Isolation Forest has no ground truth on real data unless humans review flagged observations.
- Column mapping for real datasets is keyword based - check the printed mapping.
- Joining pest and weather data needs shared date/location keys; if they do not match, they are kept separate.
- Models saved on one machine may not load on another with a different scikit-learn version; retrain on the target.
- The dashboard replay runs inside the request, so very large replays should use the CLI.

## More documentation

`docs/DEPLOY_LINUXONE.md` | `docs/TROUBLESHOOTING.md` | `docs/DATA_SOURCES.md` | `docs/DEMO_GUIDE.md`
