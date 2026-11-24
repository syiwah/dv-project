# 🚇 MetroCare — Predicting Air Compressor Failures in Urban Metro Trains

**Survivors.py** · TFB3133/TEB3133 Data Visualization · Universiti Teknologi PETRONAS · September 2026

MetroCare turns 1.5 million readings from the Air Production Unit (APU) of a Metro do Porto train
([MetroPT-3](https://doi.org/10.24432/C5VW3R)) into early warnings of air-leak failures, and presents them
in an interactive Streamlit dashboard for maintenance teams.

## Pages

| Page | What it does | Objective |
|---|---|---|
| **Overview** | Status (normal / watch / alert), risk gauge, duty cycle, latest readings; slider to replay any past moment | 3 |
| **Sensor Explorer** | Any signal over time with failures shaded · normal vs failure violins · correlation heatmaps · scatter · hour × day heatmap · CSV export | 1 |
| **Failure Risk** | Risk timeline with your threshold, alert episodes classified as early warning vs false alarm, model overlay | 2, 3 |
| **Past Failure Review** | Replay each of the 4 reported air leaks, first-alert marker and lead time for every model | 2, 3 |
| **Why this alert?** | SHAP waterfall for any moment, push per sensor, global SHAP summary and feature importance | 3 |
| **Model Performance** | Precision, recall, F1, PR-AUC, false alarms/day, lead time · PR curves · confusion matrix · labelling comparison · limits | 2 |
| **Upload & Predict** | Upload a new sensor CSV, get minute-by-minute risk and download the predictions | 3 |
| **About & Data** | Background, pipeline diagram, data-quality report, team, references | — |

## Run it

```bash
pip install -r requirements.txt
python prepare_data.py      # load, check, clean → data/metropt3_1min.parquet (+ quality report)
python train_models.py      # features, 4 models, chronological evaluation, SHAP → models/
streamlit run app.py
```

`prepare_data.py` looks for the MetroPT-3 CSV in `data/raw/` (download from UCI, dataset 791).
The processed files in `data/` and `models/` are small enough to commit, so Streamlit Community Cloud
can run `app.py` directly without the 208 MB raw file.

## Method (proposal §4.2)

1. **Load & check** — timestamps parsed; duplicates, gaps, constant columns and negative values counted.
2. **Clean** — duplicates removed, gaps ≤ 5 min forward-filled, spikes *flagged not deleted*.
3. **Label** — minutes inside a failure report = 1; a second scheme also labels the 2 h before each failure.
4. **Explore** — Sensor Explorer page.
5. **Features** — 10-minute and 1-hour mean, std, min, max and trend for analogue sensors; on-time share for digital ones; compressor **duty cycle** (share of time motor current > 1 A).
6. **Models** — Logistic Regression, Random Forest, Gradient Boosting (balanced class weights) and an unsupervised Isolation Forest.
7. **Fair test & explain** — train Feb–May (failures 1–2), test Jun–Sep (failures 3–4), never shuffled; scaling fitted on training data only. Metrics: precision, recall, F1, PR-AUC, false alarms/day, lead time. SHAP and feature importance explain alerts.

For the timeline views every minute is scored **out of sample**: Jun–Sep by the model trained on Feb–May, and
Feb–May by a model trained on Jun–Sep. Reported metrics use only the forward direction.

## Results (test period Jun–Sep 2020, failures 3–4, alert threshold 0.5)

| Model | Precision | Recall | F1 | PR-AUC | False alarms/day | Test failures caught |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.53 | 0.62 | 0.57 | 0.609 | 0.20 | 1/2 |
| Random Forest | 0.00 | 0.00 | 0.00 | 0.189 | 0.04 | 0/2 |
| Gradient Boosting | 1.00 | 0.19 | 0.32 | 0.418 | 0.00 | 1/2 |
| Isolation Forest | 0.07 | 0.09 | 0.08 | 0.233 | 0.17 | 2/2 |
| **Hybrid (LogReg + IsoForest)** ★ | 0.28 | 0.70 | 0.40 | 0.606 | 0.33 | 2/2 |

**Key findings**
- During every failure the compressor runs almost nonstop (duty cycle ≈ 100% vs ≈ 40% normally), oil temperature rises 10–16 °C and TP3 pressure falls.
- **Failure 4 had a different signature** (DV_pressure stayed near zero; the low-pressure switch fired instead), so supervised models trained on failures 1–2 missed it. Isolation Forest caught it.
- The **hybrid** (higher of Logistic Regression and Isolation Forest) catches both unseen failures with PR-AUC ≈ 0.61 and about one false alarm every three days. It was added after seeing the single-model results, so its test scores are slightly optimistic; it is also the best model in the reverse check (train Jun–Sep, test Feb–May).
- Alerts fire at about the time a failure starts, not hours before; a Watch-level warning appears about 1 h before failure 4, as oil temperature climbs.

## Limits

Four failures, all air leaks, from one train. Treat results as a **proof of concept**, not a deployable safety system.

## Project layout

```
app.py                 Streamlit entry point (navigation + sidebar controls)
views/                 one file per page
metrocare/             config, features, models, scoring, cached data loaders, charts, UI theme
prepare_data.py        steps 1–2
train_models.py        steps 3–7
data/  models/         generated artifacts
```
