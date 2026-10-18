"""AgroGuard AI dashboard. Run with:  streamlit run app.py"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from config import settings
from src import demo_data, evaluate
from src.cleaning import clean_all
from src.data_discovery import discover_datasets
from src.data_quality import build_inventory, save_inventory
from src.database import Database
from src.event_processor import EventProcessor, replay_dataframe
from src.inference import Scorer
from src.schema_mapping import to_canonical
from src.train_anomaly import InsufficientDataError, load_training_frame, train_anomaly_model

st.set_page_config(page_title="AgroGuard AI", page_icon="🌾", layout="wide")
settings.ensure_dirs()


@st.cache_resource
def get_db() -> Database:
    return Database()


db = get_db()
scorer = Scorer()

SEV_ORDER = {"high": 0, "medium": 1, "elevated": 1, "low": 2}

st.sidebar.title("🌾 AgroGuard")
st.sidebar.caption("Your farm, your field updates")

main_page = st.sidebar.radio(
    "Navigate",
    [
        "Farmer Dashboard",
        "My Farm",
        "Field Monitoring",
        "Farmer Alerts",
        "Technical Console",
    ],
)

if main_page == "Technical Console":
    page = st.sidebar.selectbox(
        "Technical tools",
        [
            "Overview",
            "Data Management",
            "Event Monitoring",
            "Anomaly Detection",
            "Pest-Risk Monitoring",
            "Alerts",
            "Model Evaluation",
        ],
    )
else:
    page = main_page
st.sidebar.caption("Anomaly = unusual observation, not proof of pests. Replay/demo data is not a live sensor feed.")
counts = db.counts()
if counts["synthetic_events"]:
    st.sidebar.warning(f"{counts['synthetic_events']} stored events are SYNTHETIC demo data.")


def banner_models() -> None:
    if not scorer.anomaly_ready:
        st.warning("No anomaly model found. Go to **Data Management** and train one "
                   "(or run `python -m src.train_anomaly`).")
    elif scorer.anomaly_meta.get("data_origin") == "synthetic":
        st.info("The anomaly model was trained on SYNTHETIC demo data. Results demonstrate the pipeline only.")


# ------------------------------------------------------------ Farmer Dashboard
if page == "Farmer Dashboard":
    st.title("🌱 Your Farm at a Glance")
    st.write("Understand your field observations and what to check next.")

    latest = db.query_df("""
        SELECT e.timestamp, e.location, e.crop, e.temperature, e.humidity,
               e.rainfall, e.source, a.is_anomaly, a.severity
        FROM events e
        LEFT JOIN anomaly_results a USING(event_id)
        ORDER BY e.timestamp DESC
        LIMIT 1
    """)

    alerts = db.query_df("""
        SELECT COUNT(*) AS n
        FROM alerts
        WHERE lower(status) IN ('open', 'acknowledged')
    """)

    open_alerts = int(alerts.iloc[0]["n"]) if not alerts.empty else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Field observations", int(counts["events"]))
    c2.metric("Warnings to review", open_alerts)
    c3.metric("Unusual observations", int(counts["anomalies"]))

    if not scorer.anomaly_ready:
        st.warning(
            "Environmental monitoring is unavailable because the anomaly model "
            "has not been loaded. No risk result is being invented."
        )
    elif scorer.anomaly_meta.get("data_origin") == "synthetic":
        st.warning(
            "The model uses synthetic demonstration data. Do not use its results "
            "to make real farm-treatment decisions."
        )

    st.subheader("Latest field observation")

    if latest.empty:
        st.info(
            "No observations have been processed yet. Your project team can "
            "process historical records using Technical Console → Event Monitoring."
        )
    else:
        row = latest.iloc[0]

        left, right = st.columns(2)

        with left:
            st.write(f"**Recorded:** {row['timestamp']}")
            st.write(f"**Location:** {row['location'] or 'Not recorded'}")
            st.write(f"**Crop:** {row['crop'] or 'Not recorded'}")
            st.caption(f"Source: {row['source'] or 'Not recorded'}")

        with right:
            for label, column, unit in [
                ("Temperature", "temperature", "°C"),
                ("Humidity", "humidity", "%"),
                ("Rainfall", "rainfall", "mm"),
            ]:
                value = row[column]
                display = f"{float(value):.1f} {unit}" if pd.notna(value) else "Not recorded"
                st.write(f"**{label}:** {display}")

        st.divider()

        if pd.isna(row["is_anomaly"]):
            st.info("This observation has not been assessed by the anomaly model.")
        elif int(row["is_anomaly"]) == 1:
            st.warning(
                "Unusual environmental conditions were flagged. This does not "
                "confirm pests. Inspect leaves and stems for visible damage "
                "or pest activity."
            )
        else:
            st.success(
                "The model did not flag this observation as unusual. This does "
                "not guarantee that the crop is pest-free."
            )

    st.subheader("What you can do next")

    a, b, c = st.columns(3)

    with a:
        st.markdown("### 🔎 Inspect")
        st.write("Check leaves, stems, and the underside of leaves for visible damage.")

    with b:
        st.markdown("### 📝 Record")
        st.write("Record the date, crop, location, and any pests or symptoms you actually observe.")

    with c:
        st.markdown("### 👩‍🌾 Get advice")
        st.write("If damage is spreading, seek advice from a local agricultural expert.")

    if not scorer.risk_ready:
        st.info(
            "Validated pest-risk prediction is currently unavailable. Environmental "
            "anomalies must not be presented as confirmed pest outbreaks."
        )

# --------------------------------------------------------------- My Farm
elif page == "My Farm":
    st.title("🌾 My Farm")
    st.write("Enter your farm details for this browser session.")

    with st.form("farm_profile_form"):
        farm_name = st.text_input("Farm name")
        location = st.text_input("Farm location")

        crop = st.selectbox(
            "Main crop",
            [
                "Rice / Paddy",
                "Maize",
                "Cotton",
                "Sugarcane",
                "Vegetables",
                "Pulses",
                "Other",
            ],
        )

        farm_area = st.number_input(
            "Farm area in acres",
            min_value=0.0,
            value=0.0,
            step=0.25,
        )

        submitted = st.form_submit_button("Save farm details")

    if submitted:
        st.session_state["farm_profile"] = {
            "name": farm_name,
            "location": location,
            "crop": crop,
            "area": farm_area,
        }
        st.success("Farm details saved successfully!")

    profile = st.session_state.get("farm_profile")

    if profile:
        st.subheader("Your farm profile")
        st.write(f"**Farm:** {profile['name'] or 'Not provided'}")
        st.write(f"**Location:** {profile['location'] or 'Not provided'}")
        st.write(f"**Main crop:** {profile['crop']}")
        st.write(f"**Farm area:** {profile['area']:g} acres")
        st.caption("These details are saved for this browser session only.")

# --------------------------------------------------------- Field Monitoring
elif page == "Field Monitoring":
    st.title("🌦️ Field Monitoring")
    st.caption("Processed observations, not live sensor readings.")

    recent = db.query_df("""
        SELECT timestamp AS "Recorded at",
               location AS "Location",
               crop AS "Crop",
               temperature AS "Temperature (°C)",
               humidity AS "Humidity (%)",
               rainfall AS "Rainfall (mm)",
               source AS "Source"
        FROM events
        ORDER BY timestamp DESC
        LIMIT 100
    """)

    if recent.empty:
        st.info("No observations are available yet.")
    else:
        recent = recent.copy()

        if "Recorded at" in recent.columns:
            raw_time = recent["Recorded at"]
            parsed_time = pd.to_datetime(raw_time, errors="coerce", utc=True)

            if parsed_time.notna().any():
                recent["Recorded at"] = parsed_time.dt.strftime(
                    "%d %b %Y, %I:%M %p UTC"
                ).fillna(raw_time.astype(str))
            else:
                recent["Recorded at"] = raw_time.astype(str)

        for column in ["Temperature (°C)", "Humidity (%)", "Rainfall (mm)"]:
            if column in recent.columns:
                recent[column] = pd.to_numeric(
                    recent[column], errors="coerce"
                ).round(2)

        st.subheader("Recent field observations")

        for _, record in recent.iterrows():
            recorded_at = str(record["Recorded at"])
            location = str(record["Location"])
            crop = str(record["Crop"])

            with st.expander(
                f"{recorded_at} | {location} | {crop}"
            ):
                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Temperature",
                    f"{float(record['Temperature (°C)']):.2f} °C"
                )
                col2.metric(
                    "Humidity",
                    f"{float(record['Humidity (%)']):.2f}%"
                )
                col3.metric(
                    "Rainfall",
                    f"{float(record['Rainfall (mm)']):.2f} mm"
                )

                st.write(f"**Location:** {location}")
                st.write(f"**Crop:** {crop}")
                st.write(f"**Source:** {record['Source']}")

        st.download_button(
            "Download observations",
            recent.to_csv(index=False).encode("utf-8"),
            file_name="agroguard_observations.csv",
            mime="text/csv",
        )

# ------------------------------------------------------------ Farmer Alerts
elif page == "Farmer Alerts":
    st.title("🔔 Farm Warnings")
    alerts = db.query_df("SELECT * FROM alerts ORDER BY timestamp DESC")

    if alerts.empty:
        st.success("There are no saved warnings at the moment.")
    else:
        active = alerts[
            alerts["status"].astype(str).str.lower().isin(["open", "acknowledged"])
        ]

        st.metric("Warnings needing attention", len(active))

        for _, alert in alerts.head(30).iterrows():
            severity = str(alert.get("severity", "")).lower()
            heading = {
                "high": "Needs attention",
                "medium": "Monitor conditions",
                "elevated": "Monitor conditions",
                "low": "Routine observation",
            }.get(severity, "Field warning")

            with st.expander(
                f"{heading} | {alert.get('timestamp', '')} | "
                f"{alert.get('location') or 'Location not recorded'}"
            ):
                st.write(
                    f"**Category:** "
                    f"{str(alert.get('category', 'Unknown')).replace('_', ' ').title()}"
                )
                st.write(f"**Status:** {alert.get('status', 'Unknown').title()}")
                st.write(
                    f"**Suggested next step:** "
                    f"{alert.get('recommendation') or 'Inspect the crop and record visible symptoms.'}"
                )
                st.caption(
                    "A system warning is not proof of a pest outbreak. "
                    "Verify conditions in the field before taking action."
                )

# ------------------------------------------------------------- Technical pages
elif page == "Overview":
    st.title("Overview")
    banner_models()
    c = st.columns(5)
    c[0].metric("Events processed", counts["events"])
    c[1].metric("Anomalies", counts["anomalies"])
    c[2].metric("Open alerts", counts["alerts_open"])
    c[3].metric("Processing errors", counts["errors"])
    c[4].metric("High-risk cases", counts["risk_predictions"] if scorer.risk_ready else "n/a")
    if not scorer.risk_ready:
        st.caption("Pest-risk estimation is unavailable: no validated labelled pest data has been used to train a model.")
    disc = discover_datasets()
    st.subheader("Dta-quality status")
    if disc.found_any:
        st.success(f"{len(disc.files)} dataset file(s) found in `{disc.searched_dir}`.")
    else:
        st.warning("No real datasets found. See **Data Management** for how to add them.")
    if counts["events"] == 0:
        st.info("No events processed yet. Open **Event Monitoring** to replay data.")
    else:
        alerts = db.query_df("SELECT category, severity, COUNT(*) AS n FROM alerts GROUP BY category, severity")
        if not alerts.empty:
            st.plotly_chart(px.bar(alerts, x="category", y="n", color="severity", title="Alerts by category"),
                            use_container_width=True)

# ------------------------------------------------------------ Data Management
elif page == "Data Management":
    st.title("Data Management")
    disc = discover_datasets()
    st.write(f"**Working directory:** `{disc.working_dir}`  \n**Searched folder:** `{disc.searched_dir}`")
    if not disc.found_any:
        st.warning("No dataset files found.")
        for m in disc.messages:
            st.write("- " + m)
    else:
        st.dataframe(pd.DataFrame(disc.files)[["relative_path", "type", "size_bytes"]], use_container_width=True)
    if disc.inaccessible:
        st.error("Inaccessible files:")
        st.dataframe(pd.DataFrame(disc.inaccessible))


    st.subheader("Quality report and cleaning")
    b1, b2 = st.columns(2)
    if b1.button("Generate quality report"):
        inv = build_inventory(discovery=disc)
        save_inventory(inv)
        st.success(f"Report saved for {inv['dataset_count']} dataset(s).")
    if b2.button("Clean datasets (raw files are never changed)"):
        res = clean_all()
        st.success("Cleaning finished." if res["datasets_found"] else "Nothing to clean: no datasets found.")
    inv_path = Path(settings.REPORTS_DIR) / "data_inventory.json"
    if inv_path.exists():
        inv = json.loads(inv_path.read_text(encoding="utf-8"))
        for ds in inv.get("datasets", []):
            with st.expander(f"{ds['name']} ({ds.get('rows', 0)} rows)"):
                if ds.get("columns"):
                    st.dataframe(pd.DataFrame(ds["columns"]), use_container_width=True)
                for w in ds.get("warnings", []):
                    st.warning(w)
    for fname in ("data_quality_report.md", "cleaning_report.md", "model_evaluation_report.md"):
        p = Path(settings.REPORTS_DIR) / fname
        if p.exists():
            st.download_button(f"Download {fname}", p.read_bytes(), file_name=fname)

    st.subheader("Train anomaly model")
    st.caption("Uses cleaned real weather data if available; otherwise SYNTHETIC demo data (clearly labelled).")
    if st.button("Train / retrain anomaly model"):
        try:
            df, prov = load_training_frame()
            meta = train_anomaly_model(df, prov)
            db.record_model_run("anomaly", meta)
            st.success(f"Trained {meta['model_version']} on {meta['n_training_rows']} {meta['data_origin']} rows. "
                       "Reload the page to use it.")
        except InsufficientDataError as exc:
            st.error(f"Cannot train: {exc}")

# ---------------------------------------------------------- Event Monitoring
elif page == "Event Monitoring":
    st.title("Event Monitoring")
    st.caption("Replays historical or synthetic records one by one through the pipeline. "
               "This is a replay simulation, not a live sensor network.")
    banner_models()
    mode = st.radio("Event source", ["Synthetic demo events", "Cleaned real dataset (replay)"], horizontal=True)
    limit = st.slider("Number of events to replay", 10, 1000, 200, 10)
    frame, source = None, "synthetic"
    if mode.startswith("Synthetic"):
        frame = demo_data.generate_demo_events().drop(columns=["injected_anomaly"])
        st.write("Source: **SYNTHETIC** (generated, not real observations).")
    else:
        files = sorted(Path(settings.CLEANED_DIR).glob("*_cleaned*.csv"))
        if not files:
            st.info("No cleaned datasets yet. Clean your real datasets in **Data Management** first.")
        else:
            choice = st.selectbox("Cleaned file", [f.name for f in files])
            canon, mapping = to_canonical(pd.read_csv(Path(settings.CLEANED_DIR) / choice))
            st.write("Detected column mapping (please verify):", mapping)
            if "timestamp" not in canon.columns:
                st.error("No date column detected; cannot replay this file.")
            else:
                frame, source = canon, "replay"
    if frame is not None and st.button("Start replay", disabled=not scorer.anomaly_ready):
        proc = EventProcessor(db=db, scorer=scorer)
        bar, status = st.progress(0.0), st.empty()
        n = min(limit, len(frame))

        def _p(i: int, res: dict) -> None:
            bar.progress(i / n)
            status.write(f"Processed {i}/{n} - last: {res['status']} ({res.get('category', '')})")

        summary = replay_dataframe(frame, proc, source=source, limit=limit, on_progress=_p)
        st.success(f"Done: {summary}")
    st.subheader("Recent events")
    recent = db.query_df("SELECT event_id, timestamp, location, crop, temperature, humidity, rainfall, source "
                         "FROM events ORDER BY timestamp DESC LIMIT 100")
    if recent.empty:
        st.info("No events stored yet.")
    else:
        st.dataframe(recent, use_container_width=True)

# --------------------------------------------------------- Anomaly Detection
elif page == "Anomaly Detection":
    st.title("Anomaly Detection")
    st.caption("A low score means the observation is unusual compared with training data. "
               "It does NOT mean pests are present.")
    df = db.query_df("""SELECT e.timestamp, e.location, e.temperature, e.humidity, e.rainfall, e.source,
                               a.score, a.is_anomaly, a.severity, a.top_features
                        FROM events e JOIN anomaly_results a USING(event_id) ORDER BY e.timestamp""")
    if df.empty:
        st.info("No anomaly results yet. Replay events in **Event Monitoring**.")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        fig = px.scatter(df, x="timestamp", y="score", color="severity", hover_data=["location"],
                         color_discrete_map={"none": "#9aa0a6", "medium": "#f29900", "high": "#d93025"},
                         title="Anomaly score over time (lower = more unusual)")
        st.plotly_chart(fig, use_container_width=True)
        metric = st.selectbox("Environmental trend", ["temperature", "humidity", "rainfall"])
        st.plotly_chart(px.line(df, x="timestamp", y=metric, color="location"), use_container_width=True)
        st.subheader("Unusual observations")
        unusual = df[df["is_anomaly"] == 1].sort_values("score").copy()
        unusual["unusual_features"] = unusual["top_features"].map(
            lambda s: ", ".join(f"{d['feature']}={d['value']:.1f} (z={d['robust_z']})" for d in json.loads(s or "[]")))
        st.dataframe(unusual.drop(columns=["top_features", "is_anomaly"]), use_container_width=True)

# ------------------------------------------------------ Pest-Risk Monitoring
elif page == "Pest-Risk Monitoring":
    st.title("Pest-Risk Monitoring")
    if not scorer.risk_ready:
        st.warning("**Pest-risk estimation is not available.** No validated pest-risk model exists because no real, "
                   "labelled historical pest data has been used for training. Anomaly scores are NOT pest-risk scores.")
        st.markdown("**Data needed:** real pest observations with date and location; a documented outbreak "
                    "definition; roughly 200+ rows with 20+ positive cases across seasons and locations. "
                    "Then run `python -m src.train_classifier --file <csv> --target <column>`.")
        for ref_file in sorted(Path(settings.CLEANED_DIR).glob("*_cleaned*.csv")):
            try:
                ref = pd.read_csv(ref_file)
            except Exception:  # noqa: BLE001
                continue
            ref_canon, _ = to_canonical(ref)
            if "pest_risk" not in ref.columns or "timestamp" not in ref_canon.columns:
                continue
            st.subheader("Dataset-supplied `pest_risk` column (informational only)")
            st.warning("This column came with your dataset. It can be reproduced almost exactly from the same day's "
                       "weather, so it looks computed by a rule rather than recorded from field observations. It is "
                       "NOT confirmed pest presence and NOT an AgroGuard prediction. Shown for context only.")
            per_month = pd.DataFrame({"month": ref_canon["timestamp"].dt.to_period("M").dt.to_timestamp(),
                                      "level": ref["pest_risk"].astype(str)})
            per_month = per_month.dropna().groupby(["month", "level"]).size().reset_index(name="days")
            st.plotly_chart(px.bar(per_month, x="month", y="days", color="level",
                                   title=f"Days per month by dataset-supplied pest_risk level ({ref_file.name})"),
                            use_container_width=True)
            break
    else:
        risk = db.query_df("""SELECT e.timestamp, e.location, r.probability, r.level FROM risk_predictions r
                              JOIN events e USING(event_id) ORDER BY e.timestamp""")
        st.caption("Scores are model outputs from historical patterns, not confirmed infestations.")
        if risk.empty:
            st.info("No risk predictions stored yet.")
        else:
            risk["timestamp"] = pd.to_datetime(risk["timestamp"])
            st.plotly_chart(px.line(risk, x="timestamp", y="probability", color="location"), use_container_width=True)
            st.dataframe(risk.groupby(["location", "level"]).size().reset_index(name="events"))

# -------------------------------------------------------------------- Alerts
elif page == "Alerts":
    st.title("Alerts")
    alerts = db.query_df("SELECT * FROM alerts ORDER BY timestamp DESC")
    if alerts.empty:
        st.info("No alerts yet. This is normal until events have been processed.")
    else:
        f1, f2, f3 = st.columns(3)
        cats = f1.multiselect("Category", sorted(alerts["category"].unique()), default=list(alerts["category"].unique()))
        sevs = f2.multiselect("Severity", sorted(alerts["severity"].unique()), default=list(alerts["severity"].unique()))
        stat = f3.multiselect("Status", ["open", "acknowledged", "resolved"], default=["open", "acknowledged"])
        view = alerts[alerts["category"].isin(cats) & alerts["severity"].isin(sevs) & alerts["status"].isin(stat)]
        st.write(f"{len(view)} alert(s)")
        st.dataframe(view[["alert_id", "timestamp", "location", "category", "severity", "status", "source",
                           "trigger_rule"]], use_container_width=True)
        if not view.empty:
            pick = st.selectbox("Inspect alert", view["alert_id"])
            row = view[view["alert_id"] == pick].iloc[0]
            st.json({"evidence": json.loads(row["evidence"] or "{}"),
                     "data_quality_warnings": json.loads(row["data_quality_warnings"] or "[]"),
                     "recommended_next_step": row["recommendation"], "source": row["source"]})
            b1, b2, b3 = st.columns(3)
            for col, label, status in ((b1, "Acknowledge", "acknowledged"), (b2, "Resolve", "resolved"),
                                       (b3, "Reopen", "open")):
                if col.button(label):
                    db.update_alert_status(pick, status)
                    st.rerun()

# ---------------------------------------------------------- Model Evaluation
elif page == "Model Evaluation":
    st.title("Model Evaluation")
    st.caption("Everything here is read from saved model metadata. Nothing is invented.")
    if st.button("Regenerate evaluation report"):
        evaluate.write_report()
    st.markdown(evaluate.build_report())
    if scorer.anomaly_meta.get("evaluation") is None and scorer.anomaly_ready:
        st.info("No reviewed observations are available, so the anomaly model has no accuracy metrics.")
