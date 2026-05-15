# Predictive Healthcare Intelligence Programme

This project develops a production‑style analytics pipeline for hospital operations, focusing on two core use‑cases:

1. **Predicting unplanned 30‑day hospital readmissions** at the point of discharge.  
2. **Forecasting emergency department (ED) demand** using multi‑year historical arrivals data.

The goal is to translate real healthcare challenges into **data‑driven decision tools** that PMs and clinical leaders can validate and use via dashboards.

---

## 1. Project objectives

The Predictive Healthcare Intelligence Programme aims to:

- **Develop predictive models for readmission and patient risk**  
  Identify patients at high risk of unplanned 30‑day readmission at discharge, enabling targeted interventions.

- **Forecast emergency department demand**  
  Build time‑series models that forecast daily ED arrivals per hospital to support staffing, bed management, and resource planning.

- **Translate healthcare challenges into data‑driven solutions**  
  Turn operational pain points (readmissions, ED crowding) into reproducible pipelines, not ad‑hoc analyses.

- **Enable PM‑led validation of model outputs**  
  Expose interpretable metrics, risk rankings, and key drivers so PMs can challenge and refine the models.

- **Build dashboards to communicate insights effectively**  
  Create Streamlit/BI dashboards that surface model outputs, risk bands, and forecasts in a non‑technical way.

---

## 2. Repository structure

```text
project-root/
├── data/
│   ├── raw/          # Original CSVs (not tracked in git)
│   ├── processed/    # Engineered tables (readmission_model_dataset.csv, ed_daily_arrivals.csv)
│   └── interim/      # Optional intermediate artefacts
├── models/           # Saved model artefacts (per MLflow run, git-ignored)
├── outputs/          # CSVs with metrics, feature importance, risk scores, forecasts
├── src/
│   ├── features/
│   │   ├── build_readmission_features.py   # Feature engineering for readmission
│   │   └── build_ed_timeseries.py          # Daily ED arrivals per hospital
│   ├── models/
│   │   ├── train_readmission_model.py      # LR/RF/XGB + thresholds + MLflow + risk scores
│   │   └── train_ed_forecast.py            # Naive + ETS ED forecasting + MLflow
│   └── utils/                              # Shared helpers/config (if any)
├── mlruns/          # MLflow local tracking directory (git-ignored)
├── .gitignore
├── README.md
└── requirements.txt
```

> **Note:** Raw and processed data, MLflow artifacts, and model binaries are intentionally git‑ignored to keep the repository lightweight and privacy‑safe.

---

## 3. Readmission risk modelling

### 3.1 Problem definition & data

- **Unit of analysis:** hospital admission (one row per admission).  
- **Target:** `unplanned_readmission_30d` (1 if unplanned readmission within 30 days, else 0).

Example columns in `data/processed/readmission_model_dataset.csv` include:

- **Identifiers & dates (not used as predictors):**  
  `admission_id`, `patient_id`, `admission_date`, `discharge_date`, `index_admission_date`, `index_discharge_date`.

- **Admission context & demographics:**  
  `admission_type`, `admission_source`, `hospital`, `ward`, `age_at_admission`, `age_band`, `gender`,  
  `charlson_comorbidity_index`, `social_support_score`.

- **Utilisation history & LOS:**  
  `length_of_stay_days`, `prior_admissions_all`, `prior_admissions_180d`, `prior_admissions_365d`,  
  `prior_ed_visits_180d`, `prior_ed_visits_365d`.

- **Clinical acuity & labs:**  
  `max_news2`, `mean_news2`, `high_news2_count`, `num_lab_results`,  
  `num_abnormal_labs`, `num_high_labs`, `abnormal_lab_ratio`.

- **Outcome‑style columns (used only for targets / QA, never as features):**  
  `readmitted_within_30d`, `readmission_reason_x`, `readmission_reason_y`,  
  `readmitted_30d`, `unplanned_readmission_30d`, `avoidable_readmission_30d`.

The overall readmission rate is ~18.7%, so this is a **moderately imbalanced classification problem**.

### 3.2 Leakage control

To keep the model deployment‑ready and clinically credible, outcome‑style columns are **explicitly excluded** from the feature set:

- Never used as predictors:  
  `readmitted_within_30d`, `readmission_reason_x`, `readmission_reason_y`,  
  `readmitted_30d`, `avoidable_readmission_30d`, `index_admission_date`, `index_discharge_date`.

Only `unplanned_readmission_30d` is used as the **target**; this avoids the “too good to be true” metrics caused by leakage.

### 3.3 Feature engineering

**Script:** `src/features/build_readmission_features.py`

Responsibilities:

- Load and merge admissions, readmissions, and related tables from `data/raw/`.  
- Construct a clean admission‑level table.  
- Create the target `unplanned_readmission_30d` and other QA flags.  
- Engineer features such as LOS bands, age bands, prior utilisation counts, NEWS2 summaries, abnormal lab ratios.  
- Save `data/processed/readmission_model_dataset.csv`.

### 3.4 Modelling pipeline (LR / RF / XGBoost)

**Script:** `src/models/train_readmission_model.py`

The script:

- Loads `readmission_model_dataset.csv`.  
- Drops leakage columns and keeps numeric/boolean predictors only.  
- Performs a stratified 80/20 train/validation split.  
- Trains and tunes three models:  
  - **Logistic Regression** (interpretable baseline).  
  - **Random Forest**.  
  - **XGBoost**.

For each model it:

- Computes metrics on the validation set:  
  - Accuracy  
  - ROC‑AUC  
  - PR‑AUC (precision–recall AUC)  
  - Precision, recall at an **F1‑optimised threshold** (not just 0.5).  
  - Precision/recall for **top‑10%** and **top‑20%** highest‑risk admissions.  
- Finds a **better decision threshold** on the validation set by scanning thresholds and optimising F1, to avoid “always predict no” behaviour.
- Logs all parameters, metrics, and artifacts to MLflow.  
- Writes:  
  - `outputs/model_comparison.csv` – metrics for all three models.  
  - `outputs/logistic_regression_coefficients.csv`.  
  - `outputs/random_forest_feature_importances.csv`.  
  - `outputs/xgboost_feature_importances.csv`.  
  - `outputs/readmission_risk_scores_xgb_val.csv` – patient‑level risk scores and bands from XGBoost.

#### Current leakage‑free performance

From `outputs/model_comparison.csv`:

| Model               | Accuracy | ROC‑AUC | PR‑AUC | Precision | Recall | Prec@Top10% | Recall@Top10% | Prec@Top20% | Recall@Top20% | Threshold_used |
|---------------------|----------|--------:|-------:|----------:|-------:|------------:|--------------:|------------:|--------------:|----------------|
| logistic_regression | 0.62     | 0.63    | 0.31   | 0.26      | 0.56   | 0.39        | 0.21          | 0.32        | 0.34          | 0.51           |
| random_forest       | 0.71     | 0.63    | 0.28   | 0.31      | 0.45   | 0.31        | 0.16          | 0.31        | 0.33          | 0.20           |
| xgboost             | 0.67     | 0.62    | 0.27   | 0.28      | 0.50   | 0.32        | 0.17          | 0.32        | 0.34          | 0.18           |

- ROC‑AUC ~0.62–0.63 and PR‑AUC ~0.27–0.31 are in line with published all‑cause 30‑day readmission models using routine EHR/claims data.
- All three models now have **non‑zero precision and recall** at their chosen thresholds; the random forest and XGBoost no longer sit at 0 recall.
- Logistic regression offers the best PR‑AUC and is the most interpretable; XGBoost provides a strong non‑linear alternative with similar ranking performance.

### 3.5 Risk scores for PM‑led validation

**File:** `outputs/readmission_risk_scores_xgb_val.csv`

This table is generated from the tuned XGBoost model on the validation set and includes:[file:67]

- `admission_id`, `patient_id`.  
- `actual_unplanned_readmission_30d`.  
- `predicted_risk` (XGBoost probability).  
- `risk_decile` (1 = top 10% highest risk, …, 10 = lowest).  
- `high_risk_10_flag`, `high_risk_20_flag`.  
- Key drivers: `age_at_admission`, `age_band`, `charlson_comorbidity_index`, `length_of_stay_days`, `prior_admissions_365d`, `prior_ed_visits_365d`, `max_news2`, `num_abnormal_labs`.

PMs and clinicians can inspect the top‑risk bands (e.g. where `high_risk_10_flag == 1`) and review whether the high‑risk cases and drivers are clinically plausible.

---

## 4. ED demand forecasting

### 4.1 ED daily arrivals dataset

**Script:** `src/features/build_ed_timeseries.py`

Source file: `data/raw/ed_visits.csv`.

Key columns:

- `ed_visit_id`, `patient_id`.  
- `arrival_datetime`, `departure_datetime`.  
- `triage_level`, `triage_category`, `chief_complaint`.  
- `day_of_week`, `hour_of_arrival`, `month`, `season`.  
- `hospital` and disposition fields such as `admitted_from_ed`, `left_ama`.

The script:

- Parses `arrival_datetime` and derives a `date` column.  
- Aggregates to **daily ED arrivals per hospital**:  
  - `date`, `hospital`, `ed_arrivals`.  
- Adds calendar features:  
  - `day_of_week` (0–6), `month`, `year`, `is_weekend`.  
- Saves `data/processed/ed_daily_arrivals.csv`.

Result: ~4,937 rows covering 2020‑01‑01 to 2024‑12‑31 across all hospitals.

### 4.2 Forecasting models (seasonal naive + ETS)

**Script:** `src/models/train_ed_forecast.py`

The script:

- Loads `ed_daily_arrivals.csv`.  
- For each hospital, builds a daily time series of `ed_arrivals`.  
- Uses **weekly seasonality** (7‑day cycle).  
- Trains two models per hospital:

1. **Seasonal naive baseline**  
   - Forecast for a given day = arrivals from the same weekday in the previous week.  
   - Evaluated on the last 7 days (hold‑out): MAE and MAPE.

2. **Exponential smoothing (ETS / Holt‑Winters)**  
   - Additive trend and additive weekly seasonality (`seasonal_periods=7`).  
   - Also evaluated on the last 7 days: MAE and MAPE.

- Logs metrics and parameters to MLflow under the `ed_demand_forecast` experiment.  
- Writes per‑hospital forecast files: `outputs/ed_forecast_<HOSPITAL>.csv`.  
- Writes a summary: `outputs/ed_forecast_summary_naive_ets.csv`.

#### Multi‑hospital forecasting performance

From `ed_forecast_summary_naive_ets.csv`:

| Hospital              | Naive MAE | Naive MAPE | ETS MAE | ETS MAPE |
|-----------------------|----------:|-----------:|--------:|---------:|
| MHN Boston General    | 0.29      | 21.4%      | 0.28    | 22.8%    |
| MHN Cambridge         | 0.57      | 50.0%      | 0.30    | 23.6%    |
| MHN Dorchester        | 1.00      | 66.7%      | 0.59    | 31.5%    |
| MHN Fenway            | 0.43      | 21.4%      | 0.49    | 30.7%    |
| MHN Jamaica Plain     | 0.43      | 28.6%      | 0.36    | 24.2%    |
| MHN Quincy            | 0.00      | 0.0%       | 0.31    | 25.8%    |
| MHN Roxbury Community | 0.14      | 14.3%      | 0.24    | 24.2%    |
| MHN South Shore       | 0.71      | 45.2%      | 0.50    | 26.5%    |

Insights:

- At some hospitals (e.g. **Cambridge, Dorchester, South Shore**), ETS **roughly halves** the error compared to the seasonal naive baseline (MAPE drops from ~45–67% to ~24–32%). 
- At others (e.g. **Boston General, Fenway, Quincy, Roxbury**), the simple weekly naive model is already very strong and ETS provides little or no benefit on the last week.
- Across sites, daily ED arrivals can often be forecast within ~20–30% relative error using simple, transparent models, which is a solid starting point for operational planning.

---

## 5. Experiment tracking (MLflow)

The project uses **MLflow** for experiment tracking:
- **Experiments:**
  - `readmission_risk` – all readmission modelling runs (LR/RF/XGB).  
  - `ed_demand_forecast` – all ED demand forecasting runs (naive & ETS per hospital).

For each run, it logs:

- Parameters: model type, hyperparameters, hospital name, season length.  
- Metrics:  
  - Readmission: accuracy, ROC‑AUC, PR‑AUC, precision, recall, prec_top_10, recall_top_10, prec_top_20, recall_top_20, threshold_used. 
  - ED: naive_mae, naive_mape, ets_mae, ets_mape.  
- Artifacts:  
  - Model comparison tables, feature importances, risk scores, per‑hospital forecast CSVs.

Launch MLflow UI locally:

```bash
mlflow ui
```

Then open the URL printed in the terminal (typically `http://127.0.0.1:5000`).

---

## 6. How to run the project locally

### 6.1 Setup

```bash
# Clone the repo
git clone https://github.com/<your-username>/predictive-healthcare-intelligence.git
cd predictive-healthcare-intelligence

# Create and activate virtual environment
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 6.2 Readmission pipeline

1. **Prepare raw data** in `data/raw/` (admissions, readmissions, etc.).  
2. Run feature engineering:

```bash
python -m src.features.build_readmission_features
```

This should create:

```text
data/processed/readmission_model_dataset.csv
```

3. Train readmission models:

```bash
python -m src.models.train_readmission_model
```

This will train Logistic Regression, Random Forest, and XGBoost, log everything to MLflow, and write:

- `outputs/model_comparison.csv`  
- `outputs/logistic_regression_coefficients.csv`  
- `outputs/random_forest_feature_importances.csv`  
- `outputs/xgboost_feature_importances.csv`  
- `outputs/readmission_risk_scores_xgb_val.csv`.

### 6.3 ED demand forecasting pipeline

1. **Prepare ED data** in `data/raw/ed_visits.csv`.
2. Build daily arrivals per hospital:

```bash
python -m src.features.build_ed_timeseries
```

This creates:

```text
data/processed/ed_daily_arrivals.csv
```

3. Run forecasting for all hospitals (naive + ETS):

```bash
python -m src.models.train_ed_forecast
```

This logs runs to MLflow and writes:

- `outputs/ed_forecast_<HOSPITAL>.csv` – actual vs predicted per hospital.  
- `outputs/ed_forecast_summary_naive_ets.csv` – MAE/MAPE per hospital and model.
---

## 7. Roadmap / Next steps

Future enhancements for the Predictive Healthcare Intelligence Programme:

1. **Readmission calibration & deployment**  
   - Calibrate probabilities (e.g. Platt / isotonic) for the chosen readmission model.  
   - Define and validate risk thresholds (e.g. top‑10%, top‑20%) with PMs and clinicians.

2. **Cohort‑specific readmission models**  
   - Build disease‑specific models (e.g. CHF, COPD, pneumonia) that often achieve higher ROC‑AUC than all‑cause models.

3. **Advanced ED forecasting and external factors**  
   - Incorporate additional features (e.g. holiday flags, weather, public events).  
   - Experiment with other models (e.g. SARIMA, Prophet, gradient boosting) where naive/ETS performance is weaker.

4. **Dashboarding (Streamlit/BI)**  
   - Readmission tab: performance summary, key drivers, high‑risk patient explorer using `readmission_risk_scores_xgb_val.csv`.  
   - ED demand tab: per‑hospital MAE/MAPE table, time‑series plots of actual vs forecast, simple what‑if views.

---

## 8. Interpretation and limitations

- All‑cause 30‑day readmission is inherently hard to predict; realistic ROC‑AUC values in the literature are often in the 0.60–0.75 range. 
- High raw accuracy (e.g. 75–80%) can be misleading in imbalanced settings and is easily achieved by predicting “no readmission” for most patients, which is why this project emphasises ROC‑AUC, PR‑AUC, top‑risk metrics, and interpretability.
- ED demand forecasts at daily resolution inevitably carry noise; models should be validated over time and potentially combined with operational judgment before driving staffing decisions.
- All models here are prototypes and should be externally validated and monitored before any production deployment.

---