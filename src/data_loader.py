import pandas as pd
from src.config import RAW_DATA_DIR

def load_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_DIR / filename)

def load_all_data():
    return {
        "patients": load_csv("patients.csv"),
        "admissions": load_csv("admissions.csv"),
        "readmissions": load_csv("readmissions.csv"),
        "ed_visits": load_csv("ed_visits.csv"),
        "vitals": load_csv("vitals.csv"),
        "lab_results": load_csv("lab_results.csv"),
        "diagnoses": load_csv("diagnoses.csv"),
        "medications": load_csv("medications.csv"),
        "data_dictionary": load_csv("data_dictionary.csv"),
    }