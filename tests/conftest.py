import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import Database  # noqa: E402


@pytest.fixture()
def db(tmp_path):
    return Database(tmp_path / "test.db")


@pytest.fixture()
def good_event():
    return {"event_id": "e1", "timestamp": "2024-06-01", "location": " field a ", "crop": "Rice",
            "temperature": 27.0, "humidity": 65.0, "rainfall": 0.0, "wind_speed": 8.0, "pressure": 1008.0,
            "source": "synthetic"}
