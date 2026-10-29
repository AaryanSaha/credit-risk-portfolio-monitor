"""Ingestion: load the 5 raw tables, fix data-quality issues, aggregate to the account grain.

Grain of the analysis table: one row per loan account (14,685 rows).
Join logic (all LEFT joins from loan_accounts so no account is ever dropped):
  loan_accounts  --customer_id-->  customers                (many accounts : 1 customer)
  loan_accounts  --customer_id-->  agg(bureau_credit)       (bureau lines aggregated per customer first)
  loan_accounts  --account_id -->  agg(repayment_history)   (12 monthly rows aggregated per account)
  loan_accounts  --account_id -->  agg(collections_events)  (events aggregated per account)
Every decision taken is appended to DQ_LOG and written to data/processed/dq_log.csv.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import RAW, PROCESSED, BUREAU_REF_DATE

DQ_LOG: list[dict] = []


def _log(table: str, issue: str, n: int, decision: str) -> None:
    DQ_LOG.append({"table": table, "issue": issue, "rows_affected": int(n), "decision": decision})


# --------------------------------------------------------------------------- load
def load_raw() -> dict[str, pd.DataFrame]:
    return {
        "customers": pd.read_csv(RAW / "customers.csv", parse_dates=["customer_since"]),
        "loans": pd.read_csv(RAW / "loan_accounts.csv", parse_dates=["origination_date"]),
        "bureau": pd.read_csv(RAW / "bureau_credit.csv", parse_dates=["date_opened", "date_closed"]),
        "repay": pd.read_csv(RAW / "repayment_history.csv"),
        "coll": pd.read_csv(RAW / "collections_events.csv", parse_dates=["event_date"]),
    }


# --------------------------------------------------------------------------- clean
def clean_customers(c: pd.DataFrame) -> pd.DataFrame:
    c = c.copy()
    # key integrity
    assert c.customer_id.is_unique
    # 1. gender labels
    n = c.gender.isin(["M", "F"]).sum()
    c["gender"] = c.gender.replace({"M": "Male", "F": "Female"})
    _log("customers", "Inconsistent gender coding ('M'/'F' vs 'Male'/'Female')", n,
         "Standardised to Male/Female. Gender is kept for profiling only and excluded from the model.")
    # 2. missing values: keep NaN + indicator, impute downstream (never drop rows)
    for col in ["years_employed", "num_dependents", "years_at_residence"]:
        n = c[col].isna().sum()
        c[f"{col}_missing"] = c[col].isna().astype(int)
        _log("customers", f"Missing {col}", n,
             "Rows kept; missing-indicator added; value imputed with median by employment_type (model) / left NaN (profiling).")
    # 3. implausible tenure: years employed > age - 16
    bad = c.years_employed > (c.age - 16)
    c["years_employed_capped_flag"] = bad.astype(int)
    c.loc[bad, "years_employed"] = (c.loc[bad, "age"] - 16).clip(lower=0)
    _log("customers", "years_employed exceeds (age - 16): implausible", bad.sum(),
         "Capped at age - 16 and flagged.")
    # 4. income floor / ceiling
    n = (c.annual_income == c.annual_income.min()).sum()
    _log("customers", f"Income floored at {c.annual_income.min():,.0f} (piled-up minimum value)", n,
         "Kept as-is (likely a reporting floor); income is log-transformed in the model.")
    # impute for modelling
    for col in ["years_employed", "num_dependents", "years_at_residence"]:
        c[col + "_imp"] = c[col].fillna(c.groupby("employment_type")[col].transform("median")).fillna(c[col].median())
    c["income_band"] = pd.cut(c.annual_income, [0, 40_000, 60_000, 80_000, 100_000, 1e9],
                              labels=["<40k", "40-60k", "60-80k", "80-100k", "100k+"]).astype(str)
    c["age_band"] = pd.cut(c.age, [0, 25, 35, 45, 55, 120], labels=["<=25", "26-35", "36-45", "46-55", "56+"]).astype(str)
    return c


def clean_loans(l: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
    l = l.copy()
    assert l.account_id.is_unique
    orphan = (~l.customer_id.isin(c.customer_id)).sum()
    _log("loan_accounts", "Accounts whose customer_id is not in customers", orphan, "None found; join is lossless.")
    n = l.interest_rate_pct.isna().sum()
    l["interest_rate_missing"] = l.interest_rate_pct.isna().astype(int)
    l["interest_rate_imp"] = l.interest_rate_pct.fillna(l.groupby("loan_type").interest_rate_pct.transform("median"))
    _log("loan_accounts", "Missing interest_rate_pct", n,
         "Imputed with median of the loan_type; indicator kept. Rate is not used to define the target.")
    l["loan_type"] = l.loan_type
    n = (l.disbursal_channel == "Partner/DSA").sum()
    _log("loan_accounts", "Channel label 'Partner/DSA' differs from dictionary ('Partner-DSA')", n,
         "Kept the data's spelling ('Partner/DSA').")
    l["origination_year"] = l.origination_date.dt.year
    return l


def clean_bureau(b: pd.DataFrame) -> pd.DataFrame:
    b = b.copy()
    assert b.bureau_id.is_unique
    n = ((b.account_status == "Written-Off") & b.date_closed.notna()).sum()
    _log("bureau_credit", "Written-Off lines carry a date_closed (and no 'Closed' lines are missing it)", n,
         "Treated date_closed on write-offs as the write-off date; status field is the source of truth.")
    n = b.days_past_due.isna().sum()
    b["dpd_missing"] = b.days_past_due.isna().astype(int)
    _log("bureau_credit", "Missing days_past_due", n,
         "Not imputed to 0 (would hide risk). Aggregates use skipna; count of unknown lines kept as a feature.")
    n = (b.outstanding_amount > b.sanctioned_amount).sum()
    _log("bureau_credit", "Outstanding > sanctioned", n, "None found.")
    return b


# --------------------------------------------------------------------------- aggregate
def agg_bureau(b: pd.DataFrame) -> pd.DataFrame:
    ref = pd.Timestamp(BUREAU_REF_DATE)
    b = b.assign(
        is_active=(b.account_status == "Active"), is_closed=(b.account_status == "Closed"),
        is_wo=(b.account_status == "Written-Off"),
        dpd30=(b.days_past_due >= 30), dpd60=(b.days_past_due >= 60),
        fintech=(b.lender_category == "Fintech"),
        active_sanc=np.where(b.account_status == "Active", b.sanctioned_amount, 0.0),
        active_out=np.where(b.account_status == "Active", b.outstanding_amount, 0.0),
        age_m=(ref - b.date_opened).dt.days / 30.4,
    )
    g = b.groupby("customer_id").agg(
        bur_n_lines=("bureau_id", "count"), bur_n_active=("is_active", "sum"), bur_n_closed=("is_closed", "sum"),
        bur_n_writeoff=("is_wo", "sum"), bur_total_sanctioned=("sanctioned_amount", "sum"),
        bur_total_outstanding=("outstanding_amount", "sum"), bur_active_sanc=("active_sanc", "sum"),
        bur_active_out=("active_out", "sum"), bur_max_dpd=("days_past_due", "max"),
        bur_mean_dpd=("days_past_due", "mean"), bur_n_dpd30=("dpd30", "sum"), bur_n_dpd60=("dpd60", "sum"),
        bur_dpd_unknown=("dpd_missing", "sum"), bur_fintech_share=("fintech", "mean"),
        bur_credit_age_m=("age_m", "max"),
    )
    g["bur_utilisation"] = (g.bur_active_out / g.bur_active_sanc.replace(0, np.nan))
    g["bur_has_writeoff"] = (g.bur_n_writeoff > 0).astype(int)
    return g.drop(columns=["bur_active_sanc"]).reset_index()


def agg_repay(r: pd.DataFrame) -> pd.DataFrame:
    r = r.sort_values(["account_id", "payment_month"]).copy()
    r["missed"] = (r.payment_status == "Missed").astype(int)
    r["late"] = (r.payment_status == "Late").astype(int)
    r["short"] = (r.emi_due - r.emi_paid).clip(lower=0)
    # longest streak of consecutive non-on-time months
    def streak(s: pd.Series) -> int:
        best = cur = 0
        for v in s.values:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        return best
    r["notontime"] = (r.payment_status != "On-Time").astype(int)
    first3 = r[r.payment_month <= 3].groupby("account_id").days_past_due.mean().rename("rep_dpd_first3")
    last3 = r[r.payment_month >= 10].groupby("account_id").days_past_due.mean().rename("rep_dpd_last3")
    last_bal = r[r.payment_month == r.payment_month.max()].set_index("account_id").outstanding_balance.rename("rep_last_balance")
    g = r.groupby("account_id").agg(
        rep_months=("payment_month", "count"), rep_dpd_mean=("days_past_due", "mean"),
        rep_dpd_max=("days_past_due", "max"), rep_n_late=("late", "sum"), rep_n_missed=("missed", "sum"),
        rep_emi_due=("emi_due", "sum"), rep_emi_paid=("emi_paid", "sum"), rep_shortfall=("short", "sum"),
        rep_emi_monthly=("emi_due", "mean"),
    ).join([first3, last3, last_bal])
    g["rep_streak_notontime"] = r.groupby("account_id").notontime.apply(streak)
    g["rep_pay_ratio"] = g.rep_emi_paid / g.rep_emi_due
    g["rep_dpd_trend"] = g.rep_dpd_last3 - g.rep_dpd_first3
    g["rep_pct_ontime"] = 1 - (g.rep_n_late + g.rep_n_missed) / g.rep_months
    return g.reset_index()


def agg_coll(e: pd.DataFrame) -> pd.DataFrame:
    sev = {"Reminder Call": 1, "SMS/Email Notice": 1, "Field Visit": 2, "Settlement Offer": 3,
           "Legal Notice": 4, "Written-Off Referral": 5}
    e = e.assign(sev=e.action_type.map(sev))
    return e.groupby("account_id").agg(
        coll_n_events=("collection_event_id", "count"), coll_recovered=("amount_recovered", "sum"),
        coll_max_severity=("sev", "max"), coll_last_event=("event_date", "max"),
    ).reset_index()


# --------------------------------------------------------------------------- build
def build_analysis_table() -> pd.DataFrame:
    DQ_LOG.clear()
    raw = load_raw()
    c = clean_customers(raw["customers"])
    l = clean_loans(raw["loans"], c)
    b = clean_bureau(raw["bureau"])
    r, e = raw["repay"], raw["coll"]

    _log("repayment_history", "Duplicate (account_id, payment_month) rows", r.duplicated(["account_id", "payment_month"]).sum(),
         "None found.")
    n_short = (r.groupby("account_id").size() != 12).sum()
    _log("repayment_history", "Accounts without exactly 12 monthly rows", n_short,
         "None: every account has 12 months regardless of tenure, so history is a fixed 12-month observation window.")
    _log("repayment_history", "payment_month is an index since origination, not a calendar month", len(r),
         "Features are computed on the index; no calendar join made. Calendar trends use origination_date only.")

    em = e.merge(l[["account_id", "origination_date"]], on="account_id")
    n = (em.event_date < em.origination_date).sum()
    _log("collections_events", "Event dated before the account's origination date", n,
         "Dates are not reliable for sequencing. Collections used only as counts/severity for reporting, never for timing.")
    _log("collections_events", "Collections activity is a consequence of delinquency/default", len(e),
         "EXCLUDED from model features to prevent leakage; used on the dashboard and for monitoring only.")

    n = (l.merge(c[["customer_id", "customer_since"]], on="customer_id").eval("customer_since > origination_date")).sum()
    _log("loan_accounts", "origination_date earlier than customer_since", n,
         "Flagged only. 'customer tenure' not used as a feature (it would be negative for these rows).")

    cb = agg_bureau(b)
    n_nb = (~c.customer_id.isin(cb.customer_id)).sum()
    _log("bureau_credit", "Customers with no external bureau lines (thin file)", n_nb,
         "Kept; bur_* counts set to 0, ratios left NaN, bur_thin_file=1.")
    n = (b.merge(l.groupby("customer_id").origination_date.min().rename("first_orig").reset_index(), on="customer_id")
         .eval("date_opened > first_orig")).sum()
    _log("bureau_credit", "Bureau lines opened AFTER the customer's first loan origination (not point-in-time)", n,
         "Bureau treated as a current snapshot (as the data dictionary implies). Sensitivity check in model.py "
         "re-scores using only lines opened before origination. Production must use as-of-origination bureau pulls.")

    df = (l.merge(c, on="customer_id", how="left", validate="m:1")
            .merge(cb, on="customer_id", how="left", validate="m:1")
            .merge(agg_repay(r), on="account_id", how="left", validate="1:1")
            .merge(agg_coll(e), on="account_id", how="left", validate="1:1"))
    assert len(df) == len(l), "join changed row count"

    cnt = [x for x in df.columns if x.startswith("bur_") and x not in
           ("bur_max_dpd", "bur_mean_dpd", "bur_utilisation", "bur_credit_age_m", "bur_fintech_share")]
    df["bur_thin_file"] = df.bur_n_lines.isna().astype(int)
    df[cnt] = df[cnt].fillna(0)
    df["coll_n_events"] = df.coll_n_events.fillna(0)
    df["coll_recovered"] = df.coll_recovered.fillna(0)
    df["coll_max_severity"] = df.coll_max_severity.fillna(0)

    # ratios
    df["loan_to_income"] = df.loan_amount / df.annual_income
    df["emi_to_income"] = df.rep_emi_monthly * 12 / df.annual_income
    df["bureau_debt_to_income"] = df.bur_total_outstanding / df.annual_income
    df["balance_ratio"] = df.rep_last_balance / df.loan_amount
    df["log_income"] = np.log(df.annual_income)
    df["log_loan_amount"] = np.log(df.loan_amount)
    df["default_label"] = df.target_default.map({1: "Defaulted", 0: "Performing"})
    return df


def save(df: pd.DataFrame) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED / "analysis_table.parquet", index=False)
    pd.DataFrame(DQ_LOG).to_csv(PROCESSED / "dq_log.csv", index=False)


if __name__ == "__main__":
    t = build_analysis_table()
    save(t)
    print(t.shape)
    print(pd.DataFrame(DQ_LOG).to_string())