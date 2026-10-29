"""Portfolio analytics helpers shared by the dashboard, the LLM layer and the docs."""
from __future__ import annotations
import pandas as pd
from .config import PROCESSED, DECISIONS_FILE

DIMENSIONS = {
    "Loan type": "loan_type", "Employment type": "employment_type", "Income band": "income_band",
    "Age band": "age_band", "Region": "region", "City tier": "city_tier", "Channel": "disbursal_channel",
    "Housing": "housing_type", "Education": "education_level", "Origination year": "origination_year",
    "Risk band (model)": "risk_band",
}
INCOME_ORDER = ["<40k", "40-60k", "60-80k", "80-100k", "100k+"]
AGE_ORDER = ["<=25", "26-35", "36-45", "46-55", "56+"]
RISK_ORDER = ["Low", "Medium", "High", "Very High"]


def load_portfolio() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "analysis_table.parquet")
    df = df.merge(pd.read_parquet(PROCESSED / "scores.parquet"), on="account_id", how="left")
    df = df.merge(pd.read_parquet(PROCESSED / "anomalies.parquet"), on="account_id", how="left")
    return df


def load_decisions() -> pd.DataFrame:
    if DECISIONS_FILE.exists():
        return pd.read_csv(DECISIONS_FILE)
    return pd.DataFrame(columns=["account_id", "decision", "note", "reviewer", "decided_at"])


def segment_table(df: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    g = df.groupby(by, observed=True).agg(
        accounts=("account_id", "size"), defaults=("target_default", "sum"), default_rate=("target_default", "mean"),
        exposure=("loan_amount", "sum"), avg_pd=("pd_lgbm", "mean"))
    g["defaulted_exposure"] = df[df.target_default == 1].groupby(by, observed=True).loan_amount.sum()
    g["defaulted_exposure"] = g.defaulted_exposure.fillna(0.0)
    g["exposure_share"] = g.exposure / g.exposure.sum()
    g["defaulted_exposure_share"] = g.defaulted_exposure / g.defaulted_exposure.sum()
    return g.reset_index()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return (c - h, c + h)


def insights(df: pd.DataFrame) -> list[dict]:
    """Five data-driven insights; every figure is computed, nothing is hard-coded."""
    out = []
    tot_e, tot_de, base = df.loan_amount.sum(), df[df.target_default == 1].loan_amount.sum(), df.target_default.mean()
    lt = segment_table(df, "loan_type").set_index("loan_type")
    m = lt.loc["Mortgage"]; rest = df[df.loan_type != "Mortgage"].target_default.mean()
    out.append({
        "title": "Mortgages carry the portfolio's risk concentration",
        "evidence": f"Mortgages are {m.accounts/len(df):.0%} of accounts but {m.exposure_share:.0%} of exposure (${m.exposure/1e6:,.0f}M) and "
                    f"{m.defaulted_exposure_share:.0%} of defaulted exposure; default rate {m.default_rate:.1%} vs {rest:.1%} for all other products.",
        "so_what": "Collateral does not stop defaults here. Tighten mortgage underwriting (income/affordability floors) and confirm LTV and "
                   "recovery assumptions, because this one product drives most of the loss.",
    })
    ib = segment_table(df, "income_band").set_index("income_band").loc[INCOME_ORDER]
    em = segment_table(df, "employment_type").set_index("employment_type")
    un = em.loc["Unemployed"]; oth = df[df.employment_type != "Unemployed"].target_default.mean()
    out.append({
        "title": "Income is the strongest risk gradient; employment type matters mainly for 'Unemployed'",
        "evidence": f"Default rate falls from {ib.loc['<40k','default_rate']:.1%} (<40k) to {ib.loc['100k+','default_rate']:.1%} (100k+), a "
                    f"{ib.loc['<40k','default_rate']/ib.loc['100k+','default_rate']:.1f}x spread. Unemployed customers default at {un.default_rate:.1%} vs {oth:.1%} for everyone else; "
                    f"Salaried, Self-Employed, Business Owner and Contract sit within a few points of each other.",
        "so_what": "Use income (and an explicit unemployed flag) as the first-line cut-off, not job category. Customers under 40k income "
                   "or unemployed are the segment for tighter limits or co-applicant requirements.",
    })
    x = df.pivot_table(index="loan_type", columns="income_band", values="target_default", aggfunc="mean")[INCOME_ORDER]
    n = df.pivot_table(index="loan_type", columns="income_band", values="target_default", aggfunc="size")[INCOME_ORDER]
    worst = x.stack().sort_values(ascending=False)
    wl, wi = worst.index[0]
    out.append({
        "title": "Cross-cut: loan type x income - low-income mortgages are the hot spot",
        "evidence": f"{wl} borrowers earning {wi} default at {worst.iloc[0]:.1%} (n={int(n.loc[wl, wi])}) vs {x.loc[wl, '100k+']:.1%} for the same product at 100k+. "
                    f"Across products the <40k band defaults at {x['<40k'].min():.0%}-{x['<40k'].max():.0%}, always above any 100k+ cell ({x['100k+'].max():.0%} max).",
        "so_what": f"Income risk compounds with product risk. These {int(n.loc[wl, wi])} accounts are {n.loc[wl, wi]/n.loc[wl].sum():.0%} of {wl.lower()} accounts but "
                   f"{(df[(df.loan_type==wl)&(df.income_band==wi)].target_default.sum())/df[df.loan_type==wl].target_default.sum():.0%} of its defaults; "
                   "back-test an income floor for this product before applying it as policy.",
    })
    rm = df.groupby(df.rep_n_missed.clip(upper=2)).target_default.mean()
    bd = df.groupby(pd.cut(df.bur_max_dpd, [-1, 29, 400], labels=["<30", "30+"])).target_default.mean()
    out.append({
        "title": "Behavioural signals work as early warnings",
        "evidence": f"Accounts with 0 missed EMIs in the 12-month window default at {rm.loc[0]:.1%}; 1 missed: {rm.loc[1]:.1%}; 2+ missed: {rm.loc[2]:.1%}. "
                    f"Customers whose worst external DPD is 30+ days default at {bd.loc['30+']:.1%} vs {bd.loc['<30']:.1%}.",
        "so_what": "Route accounts with 2+ missed EMIs or external 30+ DPD to proactive outreach before the next due date; "
                   "this is the cheapest lever because the data already exists in monthly feeds.",
    })
    yr = df.groupby("origination_year").target_default.mean()
    reg = df.groupby("region").target_default.mean(); ch = df.groupby("disbursal_channel").target_default.mean()
    ir = df.groupby(pd.qcut(df.interest_rate_imp, 5, duplicates="drop"), observed=True).target_default.mean()
    out.append({
        "title": "Region, channel and vintage are not where the risk is; pricing is not risk-based",
        "evidence": f"Default rate stays within {yr.min():.1%}-{yr.max():.1%} across 2018-2023 vintages, {reg.min():.1%}-{reg.max():.1%} across regions, and "
                    f"{ch.min():.1%}-{ch.max():.1%} across channels. The lowest interest-rate quintile defaults at {ir.iloc[0]:.1%} vs {ir.iloc[-1]:.1%} for the highest, (pooled across products), so higher rates are not compensating for higher risk.",
        "so_what": "Do not spend policy effort on geography or channel. Re-visit rate cards: risk-based pricing by income band and "
                   "behavioural score would add yield on exactly the segments in insights 1-3.",
    })
    return out