"""Anomaly detection: accounts whose repayment / bureau pattern is unusual vs (a) their peer segment and (b) their own history.

Lens 1 - PEER: each metric is converted to a robust z-score (median/MAD) inside the account's peer segment
         (loan_type x income_band; falls back to loan_type if the segment has < 40 accounts). An IsolationForest
         on the peer z-scores scores multivariate oddness.
Lens 2 - OWN HISTORY: within the account's 12 repayment months, compare the last 3 months with its own first 9
         (DPD jump, sudden stop in paying after a clean run, balance rising instead of falling).
The final score is the larger of the two lens scores (percentile-ranked); the top ANOMALY_RATE are flagged.
Every flag carries plain-English reasons so the analyst can confirm or dismiss it.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from .config import PROCESSED, RAW, SEED

ANOMALY_RATE = 0.03
PEER_METRICS = {
    "rep_dpd_mean": "average repayment DPD (days)",
    "rep_pay_ratio": "share of EMI actually paid",
    "rep_n_missed": "missed EMIs in 12 months",
    "emi_to_income": "annual EMI / income",
    "loan_to_income": "loan amount / income",
    "bur_max_dpd": "worst external DPD (days)",
    "bur_n_dpd30": "external lines 30+ DPD",
    "bur_n_writeoff": "external write-offs",
    "bur_utilisation": "external credit utilisation",
    "bureau_debt_to_income": "external debt / income",
    "bur_n_lines": "external credit lines",
}
FMT = {"rep_pay_ratio": "{:.0%}", "bur_utilisation": "{:.0%}", "emi_to_income": "{:.1%}", "loan_to_income": "{:.2f}x",
       "bureau_debt_to_income": "{:.2f}x"}


def _fmt(k: str, v: float) -> str:
    return FMT.get(k, "{:.1f}").format(v)


def robust_z(s: pd.Series, grp: pd.Series) -> pd.Series:
    med = s.groupby(grp).transform("median")
    mad = (s - med).abs().groupby(grp).transform("median") * 1.4826
    sd = s.groupby(grp).transform("std")
    scale = mad.where(mad > 1e-9, sd).where(lambda x: x > 1e-9, 1.0)
    return ((s - med) / scale).clip(-8, 8)


def own_history() -> pd.DataFrame:
    r = pd.read_csv(RAW / "repayment_history.csv").sort_values(["account_id", "payment_month"])
    early, late = r[r.payment_month <= 9], r[r.payment_month >= 10]
    e = early.groupby("account_id").agg(e_dpd=("days_past_due", "mean"), e_sd=("days_past_due", "std"),
                                        e_paid=("emi_paid", "sum"), e_due=("emi_due", "sum"),
                                        e_clean=("payment_status", lambda s: (s == "On-Time").mean()))
    l = late.groupby("account_id").agg(l_dpd=("days_past_due", "mean"), l_paid=("emi_paid", "sum"), l_due=("emi_due", "sum"),
                                       l_missed=("payment_status", lambda s: (s == "Missed").sum()))
    bal = r.pivot(index="account_id", columns="payment_month", values="outstanding_balance")
    h = e.join(l)
    h["own_dpd_jump"] = (h.l_dpd - h.e_dpd) / (h.e_sd.fillna(0) + 5.0)
    h["own_paid_drop"] = (h.e_paid / h.e_due) - (h.l_paid / h.l_due)
    h["own_sudden_stop"] = ((h.e_clean >= 0.8) & (h.l_missed >= 2)).astype(int)
    h["own_balance_change"] = (bal[12] - bal[9]) / bal[9]  # positive = balance growing instead of shrinking
    return h.reset_index()


def run() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "analysis_table.parquet")
    h = own_history()
    d = df.merge(h, on="account_id", how="left")

    seg = d.loan_type + " | " + d.income_band
    cnt = seg.map(seg.value_counts())
    seg = seg.where(cnt >= 40, d.loan_type)
    Z = pd.DataFrame({k: robust_z(d[k].astype(float), seg) for k in PEER_METRICS}, index=d.index)
    Z["bur_utilisation"] = Z.bur_utilisation.fillna(0)
    Z = Z.fillna(0)
    # directional z: only the "worse" direction counts as unusual
    worse = Z.copy()
    worse["rep_pay_ratio"] = -Z["rep_pay_ratio"]
    worse = worse.clip(lower=0)

    iso = IsolationForest(n_estimators=300, contamination=ANOMALY_RATE, random_state=SEED, n_jobs=-1).fit(Z)
    iso_score = pd.Series(-iso.score_samples(Z), index=d.index)
    peer_score = 0.5 * iso_score.rank(pct=True) + 0.5 * worse.max(axis=1).rank(pct=True)

    own_raw = (d.own_dpd_jump.clip(lower=0).fillna(0) / 3 + d.own_paid_drop.clip(lower=0).fillna(0) * 2
               + d.own_sudden_stop * 1.0 + d.own_balance_change.clip(lower=0).fillna(0) * 5)
    own_score = own_raw.rank(pct=True).where(own_raw > 0, 0)
    d["anomaly_score"] = np.maximum(peer_score, own_score).round(4)
    thr = d.anomaly_score.quantile(1 - ANOMALY_RATE)
    d["anomaly_flag"] = (d.anomaly_score >= thr).astype(int)

    def reasons(i) -> list[str]:
        out = []
        zi = worse.loc[i].sort_values(ascending=False)
        for k, v in zi.items():
            if v < 3.0 or len(out) >= 3:
                break
            med = d.loc[seg == seg[i], k].median()
            out.append(f"Peer: {PEER_METRICS[k]} = {_fmt(k, d.at[i, k])} vs peer median {_fmt(k, med)} (z={Z.at[i, k]:+.1f}, segment: {seg[i]})")
        if d.at[i, "own_sudden_stop"] == 1:
            out.append(f"Own history: paid on time in {d.at[i,'e_clean']:.0%} of months 1-9, then missed {int(d.at[i,'l_missed'])} of months 10-12")
        elif d.at[i, "own_dpd_jump"] > 2:
            out.append(f"Own history: average DPD rose from {d.at[i,'e_dpd']:.1f} (months 1-9) to {d.at[i,'l_dpd']:.1f} (months 10-12)")
        if d.at[i, "own_paid_drop"] > 0.3 and not d.at[i, "own_sudden_stop"]:
            out.append(f"Own history: EMI payment ratio fell by {d.at[i,'own_paid_drop']:.0%} in the last 3 months")
        if d.at[i, "own_balance_change"] > 0.002:
            out.append(f"Own history: outstanding balance grew {d.at[i,'own_balance_change']:.1%} from month 9 to 12 (should fall)")
        return out or ["Multivariate outlier: unusual combination of metrics (IsolationForest)"]

    flagged = d.index[d.anomaly_flag == 1]
    rs = {i: reasons(i) for i in flagged}
    d["anomaly_reasons"] = [" || ".join(rs[i]) if i in rs else "" for i in d.index]
    lens = []
    for i in d.index:
        if i not in rs:
            lens.append("")
        else:
            p = any(x.startswith("Peer") for x in rs[i]); o = any(x.startswith("Own") for x in rs[i])
            lens.append("Peer + Own history" if p and o else "Peer segment" if p else "Own history" if o else "Multivariate")
    d["anomaly_lens"] = lens
    d["peer_segment"] = seg
    out = d[["account_id", "anomaly_score", "anomaly_flag", "anomaly_lens", "anomaly_reasons", "peer_segment",
             "own_dpd_jump", "own_paid_drop", "own_sudden_stop", "own_balance_change"]]
    out.to_parquet(PROCESSED / "anomalies.parquet", index=False)
    return out


if __name__ == "__main__":
    a = run()
    f = a[a.anomaly_flag == 1]
    print(len(f), f.anomaly_lens.value_counts().to_dict())
    df = pd.read_parquet(PROCESSED / "analysis_table.parquet")[["account_id", "target_default"]]
    m = a.merge(df, on="account_id")
    print("default rate flagged vs not:", m[m.anomaly_flag == 1].target_default.mean().round(3), m[m.anomaly_flag == 0].target_default.mean().round(3))
    for r in f.sort_values("anomaly_score", ascending=False).head(4).itertuples():
        print(r.account_id, r.anomaly_score, r.anomaly_lens, r.anomaly_reasons)