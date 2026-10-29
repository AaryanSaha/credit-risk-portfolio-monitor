"""Default model: logistic-regression baseline + LightGBM, with SHAP drivers and monitoring statistics.

Design choices
* Unit of prediction: loan account. Target = target_default.
* Splits are grouped by customer_id (a customer may hold up to 3 loans) so the same person never sits in train and test.
* Excluded on purpose: gender and marital_status (protected attributes), customer_since (negative tenure issue),
  and ALL collections_* columns (collections happen AFTER delinquency -> target leakage).
* Dashboard scores are out-of-fold (5-fold, customer-grouped) so no account is scored by a model that saw its label.
"""
from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import PROCESSED, MODELS, SEED, RAW

CUSTOMER_NUM = ["age", "log_income", "years_employed_imp", "num_dependents_imp", "years_at_residence_imp"]
CUSTOMER_CAT = ["employment_type", "education_level", "housing_type", "city_tier", "region", "occupation_sector"]
LOAN_NUM = ["log_loan_amount", "interest_rate_imp", "tenure_months", "collateral_flag", "loan_to_income"]
LOAN_CAT = ["loan_type", "loan_purpose", "disbursal_channel"]
BUREAU_NUM = ["bur_n_lines", "bur_n_active", "bur_n_writeoff", "bur_has_writeoff", "bur_total_outstanding",
              "bur_max_dpd", "bur_mean_dpd", "bur_n_dpd30", "bur_n_dpd60", "bur_dpd_unknown", "bur_fintech_share",
              "bur_credit_age_m", "bur_utilisation", "bur_thin_file", "bureau_debt_to_income"]
REPAY_NUM = ["rep_dpd_mean", "rep_dpd_max", "rep_n_late", "rep_n_missed", "rep_pay_ratio", "rep_shortfall",
             "rep_streak_notontime", "rep_dpd_trend", "rep_pct_ontime", "emi_to_income", "balance_ratio"]

SETS = {
    "A. Application only": (CUSTOMER_NUM + LOAN_NUM, CUSTOMER_CAT + LOAN_CAT),
    "B. + Bureau": (CUSTOMER_NUM + LOAN_NUM + BUREAU_NUM, CUSTOMER_CAT + LOAN_CAT),
    "C. + Repayment (full)": (CUSTOMER_NUM + LOAN_NUM + BUREAU_NUM + REPAY_NUM, CUSTOMER_CAT + LOAN_CAT),
}
NUM, CAT = SETS["C. + Repayment (full)"]


def ks_stat(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def metrics(y, p) -> dict:
    auc = roc_auc_score(y, p)
    return {"auc": round(auc, 4), "gini": round(2 * auc - 1, 4), "ks": round(ks_stat(y, p), 4)}


def lr_pipe(num, cat) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat)])
    return Pipeline([("pre", pre), ("lr", LogisticRegression(C=0.3, max_iter=2000, class_weight=None))])


# Chosen by 5-fold customer-grouped CV on the TRAINING split only (grid over leaves/trees/lr/min-leaf/colsample).
# The signal in this data is weak and noisy, so the best boosted model is a very shallow, heavily regularised one.
LGB_PARAMS = dict(n_estimators=300, learning_rate=0.02, num_leaves=3, min_child_samples=60, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.5, reg_lambda=10.0, random_state=SEED, verbose=-1)


def to_lgb_frame(df: pd.DataFrame, num, cat) -> pd.DataFrame:
    X = df[num + cat].copy()
    for c in cat:
        X[c] = X[c].astype("category")
    return X


def fit_lgb(Xtr, ytr, cats_ref: pd.DataFrame | None = None) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(**LGB_PARAMS).fit(Xtr, ytr)


def align_cats(X: pd.DataFrame, ref: pd.DataFrame, cat) -> pd.DataFrame:
    X = X.copy()
    for c in cat:
        X[c] = pd.Categorical(X[c].astype(str), categories=ref[c].cat.categories)
    return X


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    e = expected.dropna(); a = actual.dropna()
    cuts = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(cuts) < 3:
        return 0.0
    cuts[0], cuts[-1] = -np.inf, np.inf
    pe = np.histogram(e, cuts)[0] / len(e); pa = np.histogram(a, cuts)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def run() -> dict:
    MODELS.mkdir(exist_ok=True)
    df = pd.read_parquet(PROCESSED / "analysis_table.parquet")
    y = df.target_default.values
    groups = df.customer_id.values
    out: dict = {"n_accounts": int(len(df)), "default_rate": round(float(y.mean()), 4)}

    # ---------------- headline hold-out (30%, grouped by customer)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED).split(df, y, groups))
    out["n_train"], out["n_test"] = int(len(tr)), int(len(te))
    out["feature_sets"] = {}
    for name, (num, cat) in SETS.items():
        lr = lr_pipe(num, cat).fit(df.iloc[tr][num + cat], y[tr])
        Xtr = to_lgb_frame(df.iloc[tr], num, cat)
        gbm = fit_lgb(Xtr, y[tr])
        Xte = align_cats(to_lgb_frame(df.iloc[te], num, cat), Xtr, cat)
        out["feature_sets"][name] = {
            "logistic_regression": metrics(y[te], lr.predict_proba(df.iloc[te][num + cat])[:, 1]),
            "lightgbm": metrics(y[te], gbm.predict_proba(Xte)[:, 1]),
        }
    out["headline"] = {"logistic_regression": out["feature_sets"]["C. + Repayment (full)"]["logistic_regression"],
                       "lightgbm": out["feature_sets"]["C. + Repayment (full)"]["lightgbm"]}

    # ---------------- robustness 1: out-of-time (train <=2021 originations, test 2022-2023)
    oot_tr = np.where(df.origination_year <= 2021)[0]; oot_te = np.where(df.origination_year >= 2022)[0]
    lr = lr_pipe(NUM, CAT).fit(df.iloc[oot_tr][NUM + CAT], y[oot_tr])
    Xo = to_lgb_frame(df.iloc[oot_tr], NUM, CAT); g = fit_lgb(Xo, y[oot_tr])
    Xot = align_cats(to_lgb_frame(df.iloc[oot_te], NUM, CAT), Xo, CAT)
    out["out_of_time"] = {"train_n": int(len(oot_tr)), "test_n": int(len(oot_te)),
                          "logistic_regression": metrics(y[oot_te], lr.predict_proba(df.iloc[oot_te][NUM + CAT])[:, 1]),
                          "lightgbm": metrics(y[oot_te], g.predict_proba(Xot)[:, 1])}

    # ---------------- robustness 2: leakage / point-in-time sensitivities (LightGBM, hold-out)
    sens = {}
    Xtr = to_lgb_frame(df.iloc[tr], NUM, CAT); Xte = align_cats(to_lgb_frame(df.iloc[te], NUM, CAT), Xtr, CAT)
    sens["baseline (full feature set)"] = out["headline"]["lightgbm"]["auc"]
    # (a) add collections -> shows why it is excluded
    coll = ["coll_n_events", "coll_max_severity"]
    Xc_tr = pd.concat([Xtr, df.iloc[tr][coll]], axis=1); Xc_te = pd.concat([Xte, df.iloc[te][coll]], axis=1)
    sens["+ collections (excluded: post-delinquency, no lift)"] = round(roc_auc_score(y[te], fit_lgb(Xc_tr, y[tr]).predict_proba(Xc_te)[:, 1]), 4)
    # (b) point-in-time bureau: only lines opened on/before the loan's origination
    b = pd.read_csv(RAW / "bureau_credit.csv", parse_dates=["date_opened"])
    bb = df[["account_id", "customer_id", "origination_date"]].merge(b, on="customer_id")
    bb = bb[bb.date_opened <= bb.origination_date]
    pit = bb.assign(wo=(bb.account_status == "Written-Off")).groupby("account_id").agg(
        pit_n_lines=("bureau_id", "count"), pit_n_wo=("wo", "sum"), pit_max_dpd=("days_past_due", "max"))
    d2 = df[["account_id"]].merge(pit, on="account_id", how="left").fillna({"pit_n_lines": 0, "pit_n_wo": 0})
    d2["pit_has_wo"] = (d2.pit_n_wo > 0).astype(int)
    bur_drop = [c for c in BUREAU_NUM]
    num_pit = [c for c in NUM if c not in bur_drop]
    Xp = pd.concat([to_lgb_frame(df, num_pit, CAT).reset_index(drop=True), d2[["pit_n_lines", "pit_n_wo", "pit_has_wo", "pit_max_dpd"]]], axis=1)
    Xp_tr = Xp.iloc[tr]; Xp_te = align_cats(Xp.iloc[te], Xp_tr, CAT)
    sens["point-in-time bureau only"] = round(roc_auc_score(y[te], fit_lgb(Xp_tr, y[tr]).predict_proba(Xp_te)[:, 1]), 4)
    out["sensitivity_auc_lightgbm"] = sens

    # ---------------- 5-fold grouped OOF scoring for the dashboard (+ SHAP per fold on its held-out accounts)
    oof_lr = np.zeros(len(df)); oof_gbm = np.zeros(len(df))
    Xall = to_lgb_frame(df, NUM, CAT)
    shap_all = np.zeros((len(df), len(NUM) + len(CAT)))
    for k, (a, bidx) in enumerate(GroupKFold(n_splits=5).split(df, y, groups)):
        lr = lr_pipe(NUM, CAT).fit(df.iloc[a][NUM + CAT], y[a])
        oof_lr[bidx] = lr.predict_proba(df.iloc[bidx][NUM + CAT])[:, 1]
        Xa = Xall.iloc[a]; m = fit_lgb(Xa, y[a])
        Xb = align_cats(Xall.iloc[bidx], Xa, CAT)
        oof_gbm[bidx] = m.predict_proba(Xb)[:, 1]
        sv = shap.TreeExplainer(m).shap_values(Xb)
        sv = sv[1] if isinstance(sv, list) else sv
        shap_all[bidx] = sv
    out["oof"] = {"logistic_regression": metrics(y, oof_lr), "lightgbm": metrics(y, oof_gbm)}

    # ---------------- final model on all data (for scoring new accounts + global importance)
    final = fit_lgb(Xall, y)
    final_lr = lr_pipe(NUM, CAT).fit(df[NUM + CAT], y)
    joblib.dump({"lgbm": final, "lr": final_lr, "num": NUM, "cat": CAT, "cat_levels": {c: list(Xall[c].cat.categories) for c in CAT}},
                MODELS / "models.joblib")
    feat_names = NUM + CAT
    glob = pd.DataFrame({"feature": feat_names, "mean_abs_shap": np.abs(shap_all).mean(0)}).sort_values("mean_abs_shap", ascending=False)
    glob["share"] = glob.mean_abs_shap / glob.mean_abs_shap.sum()
    out["top_features"] = glob.head(15).round(4).to_dict("records")
    # LR standardised coefficients (numeric part) for the baseline
    pre = final_lr.named_steps["pre"]; names = pre.get_feature_names_out(); coefs = final_lr.named_steps["lr"].coef_[0]
    lrc = pd.DataFrame({"feature": names, "coef": coefs}).assign(abs=lambda d: d.coef.abs()).sort_values("abs", ascending=False).head(12)
    out["lr_top_coefficients"] = lrc[["feature", "coef"]].round(3).to_dict("records")

    # ---------------- per-account scores + top drivers
    scores = df[["account_id"]].copy()
    scores["pd_lgbm"] = oof_gbm; scores["pd_lr"] = oof_lr
    scores["risk_band"] = pd.cut(scores.pd_lgbm, [0, 0.10, 0.20, 0.35, 1.0], labels=["Low", "Medium", "High", "Very High"],
                                 include_lowest=True).astype(str)
    shap_df = pd.DataFrame(shap_all, columns=feat_names); shap_df.insert(0, "account_id", df.account_id.values)
    shap_df.to_parquet(PROCESSED / "shap_values.parquet", index=False)
    scores.to_parquet(PROCESSED / "scores.parquet", index=False)

    # ---------------- monitoring: PSI development (<=2021) vs recent (2022-23) + calibration by decile
    dev, rec = df.origination_year <= 2021, df.origination_year >= 2022
    mon = [{"item": "model score (OOF PD)", "psi": round(psi(scores.pd_lgbm[dev], scores.pd_lgbm[rec]), 4)}]
    for f in ["log_income", "age", "loan_to_income", "bur_max_dpd", "rep_dpd_mean", "rep_pay_ratio", "bur_n_writeoff", "interest_rate_imp"]:
        mon.append({"item": f, "psi": round(psi(df.loc[dev, f], df.loc[rec, f]), 4)})
    out["monitoring_psi_dev_vs_recent"] = mon
    dec = pd.qcut(scores.pd_lgbm, 10, labels=False, duplicates="drop")
    cal = pd.DataFrame({"decile": dec + 1, "pred": scores.pd_lgbm, "actual": y}).groupby("decile").agg(
        n=("pred", "size"), mean_pred=("pred", "mean"), actual_rate=("actual", "mean")).round(4).reset_index()
    out["calibration_deciles"] = cal.to_dict("records")
    fpr, tpr, _ = roc_curve(y, oof_gbm)
    out["roc_points"] = {"fpr": np.round(fpr[::max(1, len(fpr)//100)], 4).tolist(), "tpr": np.round(tpr[::max(1, len(tpr)//100)], 4).tolist()}
    json.dump(out, open(PROCESSED / "model_metrics.json", "w"), indent=2)
    return out


if __name__ == "__main__":
    r = run()
    for k in ["headline", "out_of_time", "oof", "sensitivity_auc_lightgbm", "monitoring_psi_dev_vs_recent"]:
        print(k, json.dumps(r[k]))
    print(json.dumps(r["feature_sets"], indent=1))
    print([ (d["feature"], d["share"]) for d in r["top_features"][:10]])