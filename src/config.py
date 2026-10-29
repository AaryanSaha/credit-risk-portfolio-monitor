"""Shared paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
DECISIONS_FILE = PROCESSED / "anomaly_decisions.csv"   # human review log (written by the dashboard)

# Bureau snapshot reference date (latest date seen in bureau_credit.csv)
BUREAU_REF_DATE = "2025-12-31"
SEED = 42

CATEGORICAL = ["employment_type", "occupation_sector", "education_level", "housing_type",
               "city_tier", "region", "loan_type", "loan_purpose", "disbursal_channel"]
