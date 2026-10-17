# Deploying to IBM LinuxONE Community Cloud

**Verification status:** none of the steps below have been run on a real LinuxONE server. They are
written from general Linux/s390x knowledge. Run the verification commands and trust their output, not this text.

## 1. Check the server first

```bash
uname -m                 # expect: s390x
python3 --version        # 3.9+ required by the code; 3.11 preferred
pip3 --version
git --version
```

If `python3 -m venv` fails, the `python3-venv` package is missing and you may not have admin rights - ask
the service owner, or use `pip install --user` instead of a venv.

## 2. Get the code (no credentials in Git)

```bash
git clone https://github.com/<you>/agroguard-ai.git
cd agroguard-ai
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-linuxone.txt
```

Use SSH keys or a GitHub personal-access token for private repos. **Never paste keys or tokens into
the repo, README or chat.**

## 3. Check dependencies actually installed

```bash
python -c "import pandas, numpy, sklearn, streamlit, plotly, joblib; print(pandas.__version__, numpy.__version__, sklearn.__version__, streamlit.__version__)"
```

s390x is **big-endian**. If a package has no wheel, pip compiles it, which can take a long time and
needs a compiler (`gcc`, Python headers). If that is impossible, report which package failed.

## 4. Data and models

Models should be **retrained on the server** (safest - avoids scikit-learn version/architecture mismatches):

```bash
mkdir -p data/raw            # copy datasets here with scp/sftp (check each dataset's license first)
python -m src.data_quality && python -m src.cleaning && python -m src.train_anomaly
python -m src.inference      # must print the model version and a typical/extreme test
python -m pytest -q          # run the suite on the server
```

Do **not** copy `.joblib` files from Windows/x86_64 unless the scikit-learn and NumPy versions match exactly;
joblib files are pickles and should only be loaded if you trust their source.

## 5. Start the dashboard

```bash
export AGROGUARD_DB_PATH=$HOME/agroguard-data/agroguard.db   # keep data outside the git checkout (optional)
streamlit run app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true
```

## 6. Access it safely (recommended: SSH tunnel)

Keep Streamlit bound to `127.0.0.1` and tunnel from your own computer:

```bash
ssh -L 8501:127.0.0.1:8501 <user>@<server-address>
```

Then open `http://localhost:8501` locally. Binding to `0.0.0.0` exposes the dashboard (which has no login)
to anyone who can reach the port - only do that if the platform requires it and you accept the risk.

## 7. Keep it running after you disconnect

```bash
nohup streamlit run app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true > streamlit.log 2>&1 &
```

(`tmux` or `screen` also work if installed.) Stop it with `pkill -f "streamlit run"`.

## 8. Database backup

```bash
python -c "from src.database import Database; print(Database().backup())"
```

## 9. Verification checklist

- [ ] `uname -m` printed `s390x` (or the actual architecture is noted)
- [ ] all imports in step 3 succeed
- [ ] `python -m pytest -q` passes on the server
- [ ] `python -m src.inference` loads the model trained on the server
- [ ] dashboard opens through the SSH tunnel and all 7 pages load

Record the real results in your own notes; do not claim success for items you did not run.
