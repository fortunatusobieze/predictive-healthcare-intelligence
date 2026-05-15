# Predictive Healthcare Intelligence Programme

This repository contains the code and dashboard for the **Predictive Healthcare Intelligence Programme**, combining:  
- A **hospital readmission risk model** that flags patients at higher risk of returning within 30 days.  
- An **Emergency Department (ED) demand forecasting model** that predicts near‑term attendances at each hospital.  

The aim is to translate historical healthcare data into practical decision‑support tools for clinicians, operational managers, and leadership.

---

## Project structure

```text
.
├── src/
│   ├── readmission/
│   │   ├── train_readmission_model.py
│   │   ├── readmission_utils.py
│   │   └── ...
│   ├── ed_forecasting/
│   │   ├── train_ed_forecast_model.py
│   │   ├── ed_forecast_utils.py
│   │   └── ...
│   └── dashboard/
│       └── app.py               # Main Streamlit dashboard entry point
├── data/
│   ├── raw/
│   ├── processed/
│   └── models_input/
├── models/
│   ├── final_readmission_model.joblib
│   ├── final_ed_forecast_model.joblib
│   └── ...
├── mlflow.db                    # Local MLflow tracking database (if committed)
├── requirements.txt
├── README.md
└── LICENSE (optional)
```

Adjust this layout to match your actual repository.

---

## Key components

### Readmission risk model

- Supervised machine‑learning model trained on historical inpatient data.  
- Outputs **probability of 30‑day readmission** and **risk bands** (e.g. Low, Medium, High).  
- Uses clinically relevant features such as age bands, admission type, comorbidities, previous admissions, length of stay, and discharge destination.  

Model training scripts live in `src/readmission/` and save the final model artefacts into `models/`.

### ED demand forecasting

- Time‑series model forecasting ED attendances for each hospital over a short horizon (e.g. days or weeks).  
- Trained on historical daily/weekly ED arrivals per hospital, capturing trends and seasonality.  
- Produces forecasts and summary metrics (expected total volume, peak days, etc.) at hospital level.  

Model training scripts live in `src/ed_forecasting/` and save the final forecasting models into `models/`.

### Streamlit dashboard

The dashboard in `src/dashboard/app.py` exposes both components:

- **Readmission Risk page**  
  - Single‑patient prediction via interactive widgets.  
  - Batch upload (CSV) for scoring multiple patients.  
  - Risk scores and risk bands, plus explanation/feature importance visualisations.

- **ED Demand Forecasting page**  
  - Hospital selector for choosing a site.  
  - Historical vs forecast chart with clear separation.  
  - Summary metrics to support staffing and capacity planning.

---

## Getting started

These instructions explain how to run the project from scratch on a fresh machine.

### 1. Prerequisites

- Python 3.9+ (3.10+ recommended).  
- Git installed.  
- (Optional) Virtual environment tool: `venv`, `conda`, or `virtualenv`.  
- Command line / terminal access.

### 2. Clone the repository

```bash
git clone https://github.com/your-github-username/predictive-healthcare-intelligence.git
cd predictive-healthcare-intelligence
```

---

## Environment setup

### 3. Create and activate a virtual environment

Using `venv`:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you encounter installation issues on certain platforms, please check the package versions in `requirements.txt` and your Python version.

---

## Data preparation

### 5. Place or generate data

This project is designed to work with synthetic or de‑identified healthcare data.  

- **Option A – Provided data**  
  - Copy the supplied CSV files into the relevant `data/` subfolders, for example:  
    - `data/raw/readmission_model_dataset.csv`  
    - `data/raw/ed_arrivals_by_hospital.csv`  

- **Option B – Generate synthetic data**  
  - If generator scripts are included (e.g. `src/readmission/generate_synthetic_data.py`), run them to create synthetic datasets:
    ```bash
    python src/readmission/generate_synthetic_data.py
    python src/ed_forecasting/generate_synthetic_data.py
    ```
  - The scripts will write output CSVs into `data/raw/` or another configured folder.

Ensure the paths used in the training scripts match where you actually store the data.

---

## Training the models

If you only want to run the dashboard using pre‑trained models in `models/`, you can skip to “Running the dashboard”.  

For a full run from scratch, including training:

### 6. Train the readmission model

From the repository root:

```bash
python src/readmission/train_readmission_model.py
```

This script will:

- Load the readmission dataset (e.g. `data/raw/readmission_model_dataset.csv`).  
- Perform preprocessing and feature engineering.  
- Train and evaluate the readmission model.  
- Save the final model artefact into `models/final_readmission_model.joblib` (or your configured path).  

Logs and metrics may also be recorded in MLflow (see below).

### 7. Train the ED demand forecasting model

```bash
python src/ed_forecasting/train_ed_forecast_model.py
```

This script will:

- Load the ED arrivals dataset (e.g. `data/raw/ed_arrivals_by_hospital.csv`).  
- Fit the forecasting model for each hospital or for a selected site.  
- Save the final forecast model(s) into the `models/` directory.  

Again, evaluation metrics can be logged to MLflow.

---

## MLflow experiment tracking (optional but recommended)

This project uses MLflow for experiment tracking. You can either:

- Configure the tracking URI inside the training scripts, for example:

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("predictive-healthcare")
```

- Or set the environment variable before running training:

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
# On Windows (PowerShell):
# $env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
```

To launch the MLflow UI:

```bash
mlflow ui
```

Then open `http://127.0.0.1:5000` in your browser to inspect runs, parameters, and metrics.

---

## Running the dashboard locally

### 8. Launch the Streamlit app

From the repository root, with the virtual environment activated:

```bash
streamlit run src/dashboard/app.py
```

This will:

- Load the final readmission and ED forecast models from the `models/` directory.  
- Start a local Streamlit server (by default at `http://localhost:8501`).  

Open the URL shown in the terminal in your browser.

### 9. Navigating the app

Once the app is running:

- **Home / Overview**  
  - Read a short introduction and motivation for the project.  

- **Readmission Risk**  
  - Use the sidebar or page controls to switch to the readmission risk page.  
  - Enter patient characteristics via the widgets and click **Predict** to get a risk score and band.  
  - Optionally upload a CSV for **batch prediction**, then download the scored output.

- **ED Demand Forecasting**  
  - Choose a hospital/site from the dropdown.  
  - View historical ED arrivals and the forecast horizon in the chart.  
  - Review summary metrics for the upcoming period (total expected volume, peak days, etc.).

---

## Deployment (e.g. Streamlit Community Cloud)

If you wish to deploy this app publicly (for examiners or reviewers):

1. Push the repository to GitHub (public or private with appropriate sharing).  
2. On Streamlit Community Cloud, create a new app and point it to:  
   - **Repository:** `your-github-username/predictive-healthcare-intelligence`  
   - **Branch:** `main`  
   - **Main file path:** `src/dashboard/app.py`  
3. Confirm that `requirements.txt` is present at the root.  
4. Deploy, then test the public URL and include it in your report.

---

## Reproducibility notes for examiners

For an examiner wishing to reproduce the project end‑to‑end:

1. Clone the repository and create a virtual environment.  
2. Install dependencies using `pip install -r requirements.txt`.  
3. Place synthetic/de‑identified datasets into `data/` as described above.  
4. Run training scripts:
   - `python src/readmission/train_readmission_model.py`  
   - `python src/ed_forecasting/train_ed_forecast_model.py`  
5. Optionally inspect MLflow runs via `mlflow ui`.  
6. Launch the dashboard with:  
   - `streamlit run src/dashboard/app.py`  

All steps should run on a standard laptop with Python 3.9+ and no special hardware.

---

## Data, privacy, and ethics

- All data used in this repository is **synthetic** or **de‑identified** for demonstration purposes.  
- No real patient‑identifiable information is included in the repository.  
- The models and dashboard are intended as **decision‑support tools** and must not replace clinical judgement.  
- Model performance depends on the quality and representativeness of the underlying data and may degrade over time if data distribution changes.

---

## License

Specify your chosen license here (e.g. MIT, Apache 2.0), or state that the project is for academic use only.

---

## Acknowledgements

- Built using Python, Streamlit, and MLflow.  
- Developed as part of the Predictive Healthcare Intelligence Programme at Amdari