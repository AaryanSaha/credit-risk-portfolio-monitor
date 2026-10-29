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

# The public GitHub repository stores the large synthetic repayment CSV in
# numbered parts to stay within connector upload limits. Reconstruct it once,
# before any pipeline or dashboard module attempts to read the raw file.
def ensure_repayment_history() -> None:
    import shutil

    target = RAW / "repayment_history.csv"
    if target.exists():
        return
    parts_dir = RAW / "repayment_parts"
    parts = sorted(parts_dir.glob("part_*.csv"))
    if not parts:
        return
    if len(parts) != 9:
        raise RuntimeError(f"Expected 9 repayment history parts, found {len(parts)}")
    with target.open("wb") as destination:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, destination)


ensure_repayment_history()
