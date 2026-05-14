import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_all_data
from src.utils import parse_dates, standardize_columns

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


DATE_COLS = {
    "admissions": ["admission_date", "discharge_date"],
    "readmissions": ["original_discharge_date", "readmission_date"],
    "patients": ["date_of_birth", "registered_date"],
    "diagnoses": ["diagnosis_date"],
    "ed_visits": ["arrival_datetime", "departure_datetime", "visit_date"],
    "vitals": ["recorded_at", "measurement_time", "vital_time"],
    "lab_results": ["result_time", "collected_at", "collection_time", "reported_at"],
}


def safe_parse(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    existing = [c for c in cols if c in df.columns]
    if existing:
        return parse_dates(df, existing)
    return df


def load_tables() -> dict[str, pd.DataFrame]:
    data = load_all_data()

    for name in data:
        data[name] = standardize_columns(data[name])
        data[name] = safe_parse(data[name], DATE_COLS.get(name, []))

    return data


def merge_base_tables(admissions: pd.DataFrame, readmissions: pd.DataFrame) -> pd.DataFrame:
    readm_cols = [
        "original_admission_id",
        "patient_id",
        "readmission_id",
        "readmission_date",
        "days_to_readmission",
        "planned_readmission",
        "avoided_if_discharged_better",
        "readmission_type",
        "readmission_reason",
        "same_diagnosis",
    ]
    readm_cols = [c for c in readm_cols if c in readmissions.columns]

    readmissions_renamed = readmissions[readm_cols].rename(
        columns={
            "original_admission_id": "admission_id",
            "patient_id": "readmission_patient_id",
        }
    )

    base = admissions.merge(readmissions_renamed, on="admission_id", how="left")

    if "patient_id_x" in base.columns and "patient_id_y" in base.columns:
        base = base.rename(
            columns={
                "patient_id_x": "patient_id",
                "patient_id_y": "readmission_patient_id",
            }
        )

    base["planned_readmission"] = base.get("planned_readmission", 0).fillna(0).astype(int)
    base["avoided_if_discharged_better"] = (
        base.get("avoided_if_discharged_better", 0).fillna(0).astype(int)
    )
    base["days_to_readmission"] = pd.to_numeric(
        base.get("days_to_readmission"), errors="coerce"
    )

    base["readmitted_30d"] = (
        base["readmission_date"].notna()
        & (base["days_to_readmission"].fillna(np.inf) <= 30)
    ).astype(int)

    base["unplanned_readmission_30d"] = (
        (base["readmitted_30d"] == 1) & (base["planned_readmission"] == 0)
    ).astype(int)

    base["avoidable_readmission_30d"] = (
        (base["readmitted_30d"] == 1)
        & (base["avoided_if_discharged_better"] == 1)
    ).astype(int)

    return base


def enrich_patient_features(base: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    patient_cols = [
        "patient_id",
        "age",
        "sex",
        "gender",
        "ethnicity",
        "charlson_comorbidity_index",
        "social_support_score",
        "num_prior_admissions",
        "date_of_birth",
        "registered_date",
    ]
    patient_cols = [c for c in patient_cols if c in patients.columns]

    if patient_cols:
        base = base.merge(
            patients[patient_cols],
            on="patient_id",
            how="left",
            suffixes=("", "_patient"),
        )

    if "age" in base.columns:
        base["age_at_admission"] = pd.to_numeric(base["age"], errors="coerce")
    elif {"date_of_birth", "admission_date"}.issubset(base.columns):
        base["age_at_admission"] = (
            (base["admission_date"] - base["date_of_birth"]).dt.days / 365.25
        )
    else:
        base["age_at_admission"] = np.nan

    base["age_band"] = pd.cut(
        base["age_at_admission"],
        bins=[0, 50, 65, 75, np.inf],
        labels=["<50", "50-64", "65-74", "75+"],
        right=False,
    )

    return base


def add_temporal_admission_features(base: pd.DataFrame) -> pd.DataFrame:
    if "length_of_stay_days" in base.columns:
        base["length_of_stay_days"] = pd.to_numeric(
            base["length_of_stay_days"], errors="coerce"
        )
        base["los_band"] = pd.cut(
            base["length_of_stay_days"],
            bins=[0, 3, 6, 11, 21, np.inf],
            labels=["0-2", "3-5", "6-10", "11-20", "21+"],
            right=False,
        )

    if "icu_admitted" in base.columns:
        base["icu_admitted"] = (
            pd.to_numeric(base["icu_admitted"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
    elif "ward" in base.columns:
        base["icu_admitted"] = (
            base["ward"].astype(str).str.contains("ICU", case=False, na=False)
        ).astype(int)
    else:
        base["icu_admitted"] = 0

    if "admission_date" in base.columns:
        base["admission_dow"] = base["admission_date"].dt.dayofweek
        base["admission_month"] = base["admission_date"].dt.month
        base["weekend_admission"] = base["admission_dow"].isin([5, 6]).astype(int)

    if "discharge_date" in base.columns:
        base["discharge_dow"] = base["discharge_date"].dt.dayofweek
        base["discharge_month"] = base["discharge_date"].dt.month
        base["weekend_discharge"] = base["discharge_dow"].isin([5, 6]).astype(int)

    if "discharge_disposition" in base.columns:
        home_labels = {"home", "home with home health"}
        base["discharged_home_flag"] = (
            base["discharge_disposition"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(home_labels)
        ).astype(int)

    return base


def add_historical_admission_features(base: pd.DataFrame) -> pd.DataFrame:
    if not {"patient_id", "admission_date"}.issubset(base.columns):
        return base

    base = base.sort_values(["patient_id", "admission_date"]).copy()
    grp = base.groupby("patient_id", sort=False)

    base["prior_admissions_all"] = grp.cumcount()

    def rolling_lookback(group: pd.DataFrame, days: int) -> pd.Series:
        dates = group["admission_date"]
        vals = []
        for current in dates:
            vals.append(
                ((dates < current) & (dates >= current - pd.Timedelta(days=days))).sum()
            )
        return pd.Series(vals, index=group.index)

    base["prior_admissions_180d"] = (
        grp.apply(lambda g: rolling_lookback(g, 180))
        .reset_index(level=0, drop=True)
    )
    base["prior_admissions_365d"] = (
        grp.apply(lambda g: rolling_lookback(g, 365))
        .reset_index(level=0, drop=True)
    )

    return base


def add_ed_history_features(base: pd.DataFrame, ed_visits: pd.DataFrame | None) -> pd.DataFrame:
    base["prior_ed_visits_180d"] = 0
    base["prior_ed_visits_365d"] = 0

    if ed_visits is None or ed_visits.empty or "patient_id" not in ed_visits.columns:
        return base

    ed = ed_visits.copy()
    ed_time_col = next(
        (c for c in ["arrival_datetime", "visit_date"] if c in ed.columns),
        None,
    )

    if ed_time_col is None or "admission_date" not in base.columns:
        return base

    ed = ed[[c for c in ["patient_id", ed_time_col] if c in ed.columns]].copy()
    ed = ed.rename(columns={ed_time_col: "ed_time"})
    ed = ed.dropna(subset=["patient_id", "ed_time"]).sort_values(["patient_id", "ed_time"])

    ed_grouped = {
        pid: g["ed_time"].to_numpy()
        for pid, g in ed.groupby("patient_id")
    }

    result_180 = []
    result_365 = []

    for _, row in base[["patient_id", "admission_date"]].iterrows():
        pid = row["patient_id"]
        adm_date = row["admission_date"]
        times = ed_grouped.get(pid)

        if times is None or pd.isna(adm_date):
            result_180.append(0)
            result_365.append(0)
            continue

        result_180.append(
            int(
                (
                    (times < adm_date.to_datetime64())
                    & (times >= (adm_date - pd.Timedelta(days=180)).to_datetime64())
                ).sum()
            )
        )
        result_365.append(
            int(
                (
                    (times < adm_date.to_datetime64())
                    & (times >= (adm_date - pd.Timedelta(days=365)).to_datetime64())
                ).sum()
            )
        )

    base["prior_ed_visits_180d"] = result_180
    base["prior_ed_visits_365d"] = result_365

    return base


def add_diagnosis_features(base: pd.DataFrame, diagnoses: pd.DataFrame | None) -> pd.DataFrame:
    base["num_diagnoses"] = 0
    base["num_unique_icd"] = 0
    base["num_poa_diagnoses"] = 0

    if diagnoses is None or diagnoses.empty or "admission_id" not in diagnoses.columns:
        return base

    dx = diagnoses.copy()
    icd_col = next((c for c in ["icd_code", "diagnosis_code"] if c in dx.columns), None)

    agg_parts = {
        "num_diagnoses": ("admission_id", "size"),
    }
    if icd_col is not None:
        agg_parts["num_unique_icd"] = (icd_col, "nunique")
    if "poa_flag" in dx.columns:
        agg_parts["num_poa_diagnoses"] = (
            "poa_flag",
            lambda s: (s.astype(str).str.upper() == "Y").sum(),
        )

    agg = dx.groupby("admission_id").agg(**agg_parts).reset_index()
    base = base.drop(columns=["num_diagnoses", "num_unique_icd", "num_poa_diagnoses"], errors="ignore")
    base = base.merge(agg, on="admission_id", how="left")

    for col in ["num_diagnoses", "num_unique_icd", "num_poa_diagnoses"]:
        if col in base.columns:
            base[col] = base[col].fillna(0)

    return base


def add_medication_features(base: pd.DataFrame, medications: pd.DataFrame | None) -> pd.DataFrame:
    base["num_medications"] = 0
    base["num_unique_medications"] = 0

    if medications is None or medications.empty or "admission_id" not in medications.columns:
        return base

    med = medications.copy()
    med_name_col = next(
        (c for c in ["medication_name", "drug_name", "med_name"] if c in med.columns),
        None,
    )

    agg_parts = {"num_medications": ("admission_id", "size")}
    if med_name_col is not None:
        agg_parts["num_unique_medications"] = (med_name_col, "nunique")

    agg = med.groupby("admission_id").agg(**agg_parts).reset_index()
    base = base.drop(columns=["num_medications", "num_unique_medications"], errors="ignore")
    base = base.merge(agg, on="admission_id", how="left")

    for col in ["num_medications", "num_unique_medications"]:
        if col in base.columns:
            base[col] = base[col].fillna(0)

    return base


def add_vitals_features(base: pd.DataFrame, vitals: pd.DataFrame | None) -> pd.DataFrame:
    base["max_news2"] = np.nan
    base["mean_news2"] = np.nan
    base["high_news2_count"] = 0
    base["num_vitals_records"] = 0

    if vitals is None or vitals.empty or "admission_id" not in vitals.columns:
        return base

    vt = vitals.copy()
    agg_parts = {"num_vitals_records": ("admission_id", "size")}

    if "news2_score" in vt.columns:
        vt["news2_score"] = pd.to_numeric(vt["news2_score"], errors="coerce")
        agg_parts["max_news2"] = ("news2_score", "max")
        agg_parts["mean_news2"] = ("news2_score", "mean")
        agg_parts["high_news2_count"] = ("news2_score", lambda s: (s >= 5).sum())

    agg = vt.groupby("admission_id").agg(**agg_parts).reset_index()
    base = base.drop(columns=["max_news2", "mean_news2", "high_news2_count", "num_vitals_records"], errors="ignore")
    base = base.merge(agg, on="admission_id", how="left")

    for col in ["high_news2_count", "num_vitals_records"]:
        if col in base.columns:
            base[col] = base[col].fillna(0)

    return base


def add_lab_features(base: pd.DataFrame, lab_results: pd.DataFrame | None) -> pd.DataFrame:
    base["num_lab_results"] = 0
    base["num_abnormal_labs"] = 0
    base["num_high_labs"] = 0
    base["num_low_labs"] = 0
    base["num_unique_lab_tests"] = 0
    base["abnormal_lab_ratio"] = np.nan

    if lab_results is None or lab_results.empty or "admission_id" not in lab_results.columns:
        return base

    labs = lab_results.copy()
    test_col = next((c for c in ["test_name", "lab_test_name"] if c in labs.columns), None)

    if "flag" in labs.columns:
        flag = labs["flag"].astype(str).str.upper()
        labs["is_high"] = (flag == "H").astype(int)
        labs["is_low"] = (flag == "L").astype(int)
        labs["is_abnormal"] = flag.isin(["H", "L"]).astype(int)
    else:
        labs["is_high"] = 0
        labs["is_low"] = 0
        labs["is_abnormal"] = 0

    agg_parts = {
        "num_lab_results": ("admission_id", "size"),
        "num_abnormal_labs": ("is_abnormal", "sum"),
        "num_high_labs": ("is_high", "sum"),
        "num_low_labs": ("is_low", "sum"),
    }
    if test_col is not None:
        agg_parts["num_unique_lab_tests"] = (test_col, "nunique")

    agg = labs.groupby("admission_id").agg(**agg_parts).reset_index()
    base = base.drop(
        columns=[
            "num_lab_results",
            "num_abnormal_labs",
            "num_high_labs",
            "num_low_labs",
            "num_unique_lab_tests",
            "abnormal_lab_ratio",
        ],
        errors="ignore",
    )
    base = base.merge(agg, on="admission_id", how="left")

    fill_zero_cols = [
        "num_lab_results",
        "num_abnormal_labs",
        "num_high_labs",
        "num_low_labs",
        "num_unique_lab_tests",
    ]
    for col in fill_zero_cols:
        if col in base.columns:
            base[col] = base[col].fillna(0)

    base["abnormal_lab_ratio"] = np.where(
        base["num_lab_results"] > 0,
        base["num_abnormal_labs"] / base["num_lab_results"],
        np.nan,
    )

    return base


def finalise_feature_table(base: pd.DataFrame) -> pd.DataFrame:
    leakage_cols = [
        "readmission_id",
        "readmission_date",
        "days_to_readmission",
        "planned_readmission",
        "avoided_if_discharged_better",
        "readmission_type",
        "readmission_reason",
        "same_diagnosis",
        "readmission_patient_id",
    ]

    feature_df = base.drop(
        columns=[c for c in leakage_cols if c in base.columns],
        errors="ignore",
    ).copy()

    if "admission_date" in feature_df.columns:
        feature_df["index_admission_date"] = feature_df["admission_date"]
    if "discharge_date" in feature_df.columns:
        feature_df["index_discharge_date"] = feature_df["discharge_date"]

    sort_cols = [c for c in ["patient_id", "admission_date"] if c in feature_df.columns]
    if sort_cols:
        feature_df = feature_df.sort_values(sort_cols).reset_index(drop=True)

    return feature_df


def build_readmission_features(output_format: str = "parquet") -> Path:
    logger.info("Loading and preparing source tables...")
    tables = load_tables()

    base = merge_base_tables(
        tables["admissions"].copy(),
        tables["readmissions"].copy(),
    )
    logger.info("Base merged table shape: %s", base.shape)

    base = enrich_patient_features(base, tables["patients"].copy())
    base = add_temporal_admission_features(base)
    base = add_historical_admission_features(base)
    base = add_ed_history_features(base, tables.get("ed_visits"))
    base = add_diagnosis_features(base, tables.get("diagnoses"))
    base = add_medication_features(base, tables.get("medications"))
    base = add_vitals_features(base, tables.get("vitals"))
    base = add_lab_features(base, tables.get("lab_results"))

    feature_df = finalise_feature_table(base)

    if output_format == "csv":
        output_path = DATA_PROCESSED_DIR / "readmission_model_dataset.csv"
        feature_df.to_csv(output_path, index=False)
    else:
        output_path = DATA_PROCESSED_DIR / "readmission_model_dataset.parquet"
        feature_df.to_parquet(output_path, index=False)

    logger.info("Saved feature dataset to %s", output_path)
    logger.info("Final feature dataset shape: %s", feature_df.shape)
    logger.info(
        "Overall 30d readmission rate: %.4f",
        feature_df["readmitted_30d"].mean(),
    )
    logger.info(
        "Unplanned 30d readmission rate: %.4f",
        feature_df["unplanned_readmission_30d"].mean(),
    )

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build readmission feature dataset")
    parser.add_argument(
        "--output-format",
        choices=["parquet", "csv"],
        default="csv"
    )
    args = parser.parse_args()
    build_readmission_features(output_format=args.output_format)