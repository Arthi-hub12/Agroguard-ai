# Hackathon demo guide (about 5 minutes)

Be upfront with judges: **events are replayed from files or synthetic**, anomalies are **unusual observations,
not pest detections**, and pest-risk prediction needs real labelled data that this demo does not have.

## Before the demo
```bash
pip install -r requirements.txt
python -m pytest -q                    # show the tests pass
python -m src.run_pipeline             # uses the dataset in data/raw/ (or: python -m src.train_anomaly && python -m src.event_processor --demo for synthetic)
streamlit run app.py
```

## Script
1. **Overview** - show the empty state, then the "synthetic data" banner. Say what is real vs synthetic.
2. **Data Management** - discovered files, quality report, cleaning (raw files untouched), downloadable reports.
   If you have a real dataset, drop it in `data/raw/` and click *Generate quality report* then *Clean datasets*.
3. **Event Monitoring** - choose synthetic demo events, replay 200. Explain validation, dedupe and persistence;
   replay again to show duplicates are skipped.
4. **Anomaly Detection** - scores over time, the unusual observations and *which* features were unusual.
5. **Alerts** - filter by severity, open evidence and the recommended next step (field inspection / expert
   confirmation - no treatment advice), acknowledge one.
6. **Pest-Risk Monitoring** - show the honest "not available" screen, the data it needs, and the labelled informational chart of the dataset's own `pest_risk` column. Explain why it is not used for a model (reproducible from same-day weather).
7. **Model Evaluation** - metrics come from saved metadata; point out the synthetic warning.

## Likely questions
- *Is this accurate?* On real data: unknown - no reviewed observations yet. Synthetic precision/recall only shows the pipeline works.
- *Why no pest prediction?* We refuse to invent labels from weather. It activates automatically when real labels are supplied.
- *Is it real-time?* It is event-driven processing over replayed records, not a sensor network.
