# Dataset sources and licenses

**Source, license and collection method of the supplied dataset are NOT yet known. Please fill in the row below.**
Fill in this table for every file you place in `data/raw/`. Do not commit raw data to GitHub unless its
license allows redistribution (`data/raw/` is git-ignored by default).

| File | What it contains | Source / URL | License | Redistribution allowed? | Date downloaded | Notes (units, coverage, known problems) |
|---|---|---|---|---|---|---|
| `AgroGuard_AI_Cleaned.csv` | Daily weather 2019-01-01 to 2024-12-31 (temperature_c, humidity_pct, rainfall_mm, wind_speed_ms, solar_radiation), `pest_risk` (0/1/2), `anomaly_label` (0/1), calendar columns; 2,192 rows, single location, no location column | _fill in_ | _fill in_ | _fill in_ | 2026-10-17 (received) | `pest_risk` is reproducible from same-day weather (looks rule-derived, not observed pests). Origin of `anomaly_label` unknown. Unknown whether measurements are real or simulated. |

## Synthetic data

`src/demo_data.py` generates **synthetic** weather-like events (`source = synthetic`). It is not derived from
any real dataset, is written to `data/processed/` (never `data/raw/`), and must not be presented as real.

## What real data would unlock

- Weather + date + location -> a meaningful anomaly model for that region.
- Pest observations (date, location, presence or count, documented outbreak definition) -> a supervised pest-risk model.
  Roughly 200+ rows with 20+ positive cases across several seasons and locations are needed before the code will train.
