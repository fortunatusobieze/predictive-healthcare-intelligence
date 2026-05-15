# src/features/build_ed_timeseries.py

from pathlib import Path
import logging

import pandas as pd

# -----------------------
# Basic setup
# -----------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ED_RAW_PATH = DATA_RAW_DIR / "ed_visits.csv"
ED_DAILY_OUTPUT_PATH = DATA_PROCESSED_DIR / "ed_daily_arrivals.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# -----------------------
# Main routine
# -----------------------

def main():
    logging.info(f"Loading ED visits from {ED_RAW_PATH}")

    df = pd.read_csv(ED_RAW_PATH)

    # Basic column checks
    required_cols = ["ed_visit_id", "arrival_datetime", "hospital"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in ed_visits.csv: {missing}")

    # Parse arrival_datetime and derive date
    logging.info("Parsing arrival_datetime and deriving date")
    df["arrival_datetime"] = pd.to_datetime(df["arrival_datetime"])
    df["date"] = df["arrival_datetime"].dt.date

    # Aggregate to daily arrivals per hospital
    logging.info("Aggregating to daily arrivals per hospital")
    daily = (
        df.groupby(["date", "hospital"])
        .agg(ed_arrivals=("ed_visit_id", "count"))
        .reset_index()
    )

    # Add simple calendar features
    logging.info("Adding calendar features")
    daily["date"] = pd.to_datetime(daily["date"])
    daily["day_of_week"] = daily["date"].dt.dayofweek  # 0=Monday
    daily["month"] = daily["date"].dt.month
    daily["year"] = daily["date"].dt.year
    daily["is_weekend"] = daily["day_of_week"].isin([5, 6]).astype(int)

    # Sort for time-series work
    daily = daily.sort_values(["hospital", "date"]).reset_index(drop=True)

    logging.info(f"Daily ED arrivals shape: {daily.shape}")
    logging.info(
        f"Date range: {daily['date'].min().date()} to {daily['date'].max().date()}"
    )

    # Save to processed
    ED_DAILY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(ED_DAILY_OUTPUT_PATH, index=False)
    logging.info(f"Saved daily ED arrivals to {ED_DAILY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()