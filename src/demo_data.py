"""Synthetic demonstration data generator.

EVERYTHING produced here is SYNTHETIC. Rows carry ``source = "synthetic"`` and
the files are written to data/processed/, never to data/raw/. Results based on
this data demonstrate how the system works; they say nothing about real-world
pest behaviour or model accuracy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import settings

LOCATIONS = {  # name -> (base temp C, base humidity %)
    "Demo Field A": (27.0, 65.0),
    "Demo Field B": (24.0, 72.0),
    "Demo Field C": (30.0, 55.0),
}
CROPS = {"Demo Field A": "rice", "Demo Field B": "cotton", "Demo Field C": "maize"}


def generate_demo_events(days: int = 180, seed: int = settings.RANDOM_SEED,
                         anomaly_rate: float = 0.03, start: str = "2024-01-01") -> pd.DataFrame:
    """Daily synthetic weather events for three demo fields, with a few injected
    unusual days (``injected_anomaly`` is for demo evaluation only)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    rows = []
    for loc, (t0, h0) in LOCATIONS.items():
        doy = dates.dayofyear.to_numpy()
        temp = t0 + 4 * np.sin(2 * np.pi * (doy - 100) / 365) + rng.normal(0, 1.2, days)
        hum = np.clip(h0 + 10 * np.sin(2 * np.pi * (doy - 180) / 365) + rng.normal(0, 4, days), 5, 100)
        rain = np.where(rng.random(days) < 0.25, rng.gamma(2.0, 4.0, days), 0.0)
        wind = np.clip(rng.normal(8, 2.5, days), 0, None)
        pres = rng.normal(1008, 3, days)
        inj = rng.random(days) < anomaly_rate
        temp = temp + inj * rng.choice([-8, 9], days)
        hum = np.clip(hum + inj * rng.choice([-30, 25], days), 0, 100)
        rain = rain + inj * rng.uniform(60, 120, days)
        for i, d in enumerate(dates):
            rows.append({
                "event_id": f"SYN-{loc[-1]}-{d:%Y%m%d}",
                "timestamp": d,
                "location": loc,
                "crop": CROPS[loc],
                "temperature": round(float(temp[i]), 2),
                "humidity": round(float(hum[i]), 2),
                "rainfall": round(float(rain[i]), 2),
                "wind_speed": round(float(wind[i]), 2),
                "pressure": round(float(pres[i]), 2),
                "source": "synthetic",
                "injected_anomaly": bool(inj[i]),
            })
    df = pd.DataFrame(rows).sort_values(["timestamp", "location"]).reset_index(drop=True)
    return df


def write_demo_file(path: Optional[Path | str] = None, **kwargs) -> Path:
    path = Path(path) if path else Path(settings.PROCESSED_DIR) / "demo_synthetic_events.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_demo_events(**kwargs).to_csv(path, index=False)
    return path


if __name__ == "__main__":
    p = write_demo_file()
    print(f"Wrote SYNTHETIC demo events to {p}")
