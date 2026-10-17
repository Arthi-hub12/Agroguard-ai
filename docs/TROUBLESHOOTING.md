# Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| "No supported files were found" | Dataset not in the searched folder, wrong path, or a `.zip` | Copy files into `data/raw/`, unzip archives, or set `AGROGUARD_RAW_DIR` (PowerShell: `$env:AGROGUARD_RAW_DIR="D:\data"`; Linux: `export AGROGUARD_RAW_DIR=/path`) |
| Dashboard says "No anomaly model found" | Model not trained | `python -m src.train_anomaly` (or the button in Data Management) |
| `Retrain with 'python -m src.train_anomaly'` | Model file from another scikit-learn/NumPy version or corrupted | Retrain on this machine |
| Replay says "No date column detected" | Column names not recognised | Rename the date column to `date` or `timestamp` in a cleaned copy, or check `src/schema_mapping.py` aliases |
| Wrong columns mapped | Keyword matching picked a different column | Read the printed "Column mapping used" line and rename columns to `temperature`, `humidity`, `rainfall`, `location` |
| Every event is "duplicate" | The same `event_id`s were replayed already | Expected - duplicates are skipped. Use a new DB path (`AGROGUARD_DB_PATH`) to start fresh |
| Many events "rejected" | Missing timestamp/weather or invalid values | Check the `processing_errors` table: `sqlite3 data/processed/agroguard.db "select * from processing_errors limit 5"` |
| Excel file not read | `openpyxl` missing | `pip install openpyxl` |
| Parquet file not read | `pyarrow` missing | `pip install pyarrow` (may be unavailable on s390x) |
| `database is locked` | Two processes writing at once | Close the other process; the code waits 30 s before failing |
| Port already in use | Another Streamlit running | Use `--server.port 8502` or `pkill -f "streamlit run"` |
| `Permission denied` on data/models | Folder owned by another user | `chmod -R u+rwX data models reports` or set the `AGROGUARD_*_DIR` variables to a folder you own |
| pip tries to compile packages (slow, errors) on LinuxONE | No s390x wheel | Use `requirements-linuxone.txt`; install compiler and headers if permitted; skip optional `pyarrow` |
| SSH tunnel connects but page is blank | Streamlit not running or different port | Check `streamlit.log` and that the ports in the `-L` option and `--server.port` match |
