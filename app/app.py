"""Credit Risk Portfolio Monitor - Streamlit dashboard (entry point: streamlit run app/app.py)."""
from __future__ import annotations
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import PROCESSED, RAW, DECISIONS_FILE
from src import analytics as an
from src import llm

st.set_page_config(page_title="Credit Risk Portfolio Monitor", page_icon=None, layout="wide")

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8d8c86"
SEQ = ["#c9defa", "#7fb1ec", "#2a78d6", "#10417f"]           # one hue, light -> dark (risk bands)
RISK_COLORS = dict(zip(an.RISK_ORDER, SEQ))
INK = "#6b6a66"


def style(fig: go.Figure, h: int = 340) -> go.Figure:
    fig.update_layout(height=h, margin=dict(l=8, r=8, t=36, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=INK, size=12), legend=dict(orientation="h", y=-0.2), title_font_size=14)
    fig.update_xaxes(gridcolor="rgba(128,128,128,.15)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,.15)", zeroline=False)
    return fig


# ----------------------------------------------------------------------------- data
def ensure_built() -> None:
    needed = ["analysis_table.parquet", "scores.parquet", "anomalies.parquet", "shap_values.parquet", "model_metrics.json"]
    if not all((PROCESSED / f).exists() for f in needed):
        with st.spinner("First run: building data, model and anomaly flags (about 1-2 minutes)..."):
            from src import build
            build.main()


@st.cache_data(show_spinner=False)
def load_all():
    df = an.load_portfolio()
    metrics = json.load(open(PROCESSED / "model_metrics.json"))
    shap_df = pd.read_parquet(PROCESSED / "shap_values.parquet").set_index("account_id")
    dq = pd.read_csv(PROCESSED / "dq_log.csv")
    return df, metrics, shap_df, dq


@st.cache_data(show_spinner=False)
def load_account_detail(account_id: str, customer_id: str):
    rep = pd.read_csv(RAW / "repayment_history.csv"); rep = rep[rep.account_id == account_id]
    bur = pd.read_csv(RAW / "bureau_credit.csv"); bur = bur[bur.customer_id == customer_id]
    col = pd.read_csv(RAW / "collections_events.csv"); col = col[col.account_id == account_id]
    return rep, bur, col


ensure_built()
df_all, metrics, shap_df, dq = load_all()
decisions = an.load_decisions()
last_dec = (decisions.sort_values("decided_at").drop_duplicates("account_id", keep="last").set_index("account_id")
            if len(decisions) else pd.DataFrame(columns=["decision", "note", "reviewer", "decided_at"]))

# ----------------------------------------------------------------------------- sidebar filters
st.sidebar.header("Filters")
st.sidebar.caption("Apply to the Portfolio, Account and Anomaly views. 'Ask the Portfolio' always queries the full portfolio.")
f = {}
f["loan_type"] = st.sidebar.multiselect("Loan type", sorted(df_all.loan_type.unique()))
f["region"] = st.sidebar.multiselect("Region", sorted(df_all.region.unique()))
f["employment_type"] = st.sidebar.multiselect("Employment type", sorted(df_all.employment_type.unique()))
f["income_band"] = st.sidebar.multiselect("Income band", an.INCOME_ORDER)
f["age_band"] = st.sidebar.multiselect("Age band", an.AGE_ORDER)
yrs = st.sidebar.slider("Origination year", 2018, 2023, (2018, 2023))
df = df_all[df_all.origination_year.between(*yrs)]
for k, v in f.items():
    if v:
        df = df[df[k].isin(v)]
st.sidebar.metric("Accounts in view", f"{len(df):,}")
if len(df) == 0:
    st.warning("No accounts match the filters."); st.stop()

st.title("Credit Risk Portfolio Monitor")
st.caption("Synthetic bank portfolio: 10,000 customers, 14,685 loan accounts. Exposure = original loan amount (USD).")

tab_port, tab_acc, tab_anom, tab_ask, tab_model, tab_dq = st.tabs(
    ["Portfolio", "Account view", "Anomaly review", "Ask the Portfolio", "Model & monitoring", "Data quality"])

# ============================================================================= PORTFOLIO
with tab_port:
    k = st.columns(6)
    de = df[df.target_default == 1].loan_amount.sum()
    k[0].metric("Accounts", f"{len(df):,}")
    k[1].metric("Exposure", f"${df.loan_amount.sum()/1e6:,.0f}M")
    k[2].metric("Default rate", f"{df.target_default.mean():.1%}", f"{df.target_default.mean()-df_all.target_default.mean():+.1%} vs portfolio", delta_color="inverse")
    k[3].metric("Defaulted exposure", f"${de/1e6:,.0f}M", f"{de/df.loan_amount.sum():.0%} of exposure", delta_color="off")
    k[4].metric("Avg model PD", f"{df.pd_lgbm.mean():.1%}")
    k[5].metric("Anomaly flags", f"{int(df.anomaly_flag.sum()):,}")

    c1, c2 = st.columns([3, 2])
    with c1:
        dim_name = st.selectbox("Break the portfolio down by", list(an.DIMENSIONS), index=0)
        dim = an.DIMENSIONS[dim_name]
        seg = an.segment_table(df, dim)
        order = {"income_band": an.INCOME_ORDER, "age_band": an.AGE_ORDER, "risk_band": an.RISK_ORDER}.get(dim)
        if order:
            seg[dim] = pd.Categorical(seg[dim], order, ordered=True); seg = seg.sort_values(dim)
        elif dim != "origination_year":
            seg = seg.sort_values("default_rate", ascending=False)
        seg[dim] = seg[dim].astype(str)
        fig = go.Figure(go.Bar(x=seg[dim], y=seg.default_rate, marker_color=BLUE, marker_cornerradius=4,
                               customdata=np.c_[seg.accounts, seg.exposure / 1e6, seg.defaulted_exposure_share],
                               hovertemplate="%{x}<br>Default rate %{y:.1%}<br>%{customdata[0]:,} accounts<br>Exposure $%{customdata[1]:,.0f}M<br>"
                                             "Share of defaulted exposure %{customdata[2]:.0%}<extra></extra>"))
        fig.add_hline(y=df.target_default.mean(), line_dash="dot", line_color=GREY, annotation_text="view average", annotation_font_color=INK)
        fig.update_yaxes(tickformat=".0%", title="Default rate"); fig.update_layout(title=f"Default rate by {dim_name.lower()}")
        st.plotly_chart(style(fig), width="stretch")
    with c2:
        show = seg[[dim, "accounts", "default_rate", "exposure", "exposure_share", "defaulted_exposure_share"]].copy()
        for c in ["default_rate", "exposure_share", "defaulted_exposure_share"]:
            show[c] = show[c] * 100
        show["exposure"] = show.exposure / 1e6
        show.columns = [dim_name, "Accounts", "Default rate", "Exposure $M", "Exposure share", "Share of defaulted exposure"]
        st.dataframe(show, hide_index=True, width="stretch", height=340, column_config={
            "Default rate": st.column_config.NumberColumn(format="%.1f%%"), "Exposure $M": st.column_config.NumberColumn(format="%.1f"),
            "Exposure share": st.column_config.NumberColumn(format="%.0f%%"), "Share of defaulted exposure": st.column_config.NumberColumn(format="%.0f%%")})
        small = seg[seg.accounts < 100]
        if len(small):
            st.caption(f"{len(small)} segment(s) have fewer than 100 accounts - read their rates with care.")

    st.subheader("Cross-dimensional cut")
    cc1, cc2, cc3 = st.columns(3)
    rows = cc1.selectbox("Rows", list(an.DIMENSIONS), index=0, key="rows")
    cols = cc2.selectbox("Columns", list(an.DIMENSIONS), index=2, key="cols")
    metric = cc3.selectbox("Metric", ["Default rate", "Exposure ($M)", "Accounts"], key="metric")
    r_, c_ = an.DIMENSIONS[rows], an.DIMENSIONS[cols]
    if r_ == c_:
        st.info("Pick two different dimensions.")
    else:
        agg = {"Default rate": ("target_default", "mean"), "Exposure ($M)": ("loan_amount", lambda s: s.sum() / 1e6), "Accounts": ("account_id", "size")}[metric]
        pv = df.pivot_table(index=r_, columns=c_, values=agg[0], aggfunc=agg[1])
        nn = df.pivot_table(index=r_, columns=c_, values="account_id", aggfunc="size")
        for ax, o in [(r_, {"income_band": an.INCOME_ORDER, "age_band": an.AGE_ORDER, "risk_band": an.RISK_ORDER}.get(r_))]:
            if o: pv, nn = pv.reindex([x for x in o if x in pv.index]), nn.reindex([x for x in o if x in nn.index])
        if an.DIMENSIONS[cols] in ("income_band", "age_band", "risk_band"):
            o = {"income_band": an.INCOME_ORDER, "age_band": an.AGE_ORDER, "risk_band": an.RISK_ORDER}[c_]
            pv, nn = pv[[x for x in o if x in pv.columns]], nn[[x for x in o if x in nn.columns]]
        txt = pv.map(lambda v: "" if pd.isna(v) else (f"{v:.0%}" if metric == "Default rate" else f"{v:,.0f}"))
        hm = go.Figure(go.Heatmap(z=pv.values, x=[str(x) for x in pv.columns], y=[str(x) for x in pv.index], colorscale=[[0, "#e8f1fc"], [1, "#10417f"]],
                                  text=txt.values, texttemplate="%{text}", customdata=nn.values,
                                  hovertemplate=f"{rows} %{{y}}<br>{cols} %{{x}}<br>{metric}: %{{text}}<br>%{{customdata:,}} accounts<extra></extra>"))
        hm.update_layout(title=f"{metric}: {rows.lower()} x {cols.lower()}")
        st.plotly_chart(style(hm, 380), width="stretch")
        st.caption("Hover a cell to see how many accounts sit behind it; cells with few accounts are noisy.")

    t1, t2 = st.columns(2)
    with t1:
        trend = df.groupby(["origination_year", "loan_type"]).target_default.mean().reset_index()
        fig = px.line(trend, x="origination_year", y="target_default", color="loan_type", markers=True,
                      color_discrete_sequence=[BLUE, ORANGE, AQUA, "#eda100", "#e87ba4"], title="Default rate by origination year and loan type")
        fig.update_yaxes(tickformat=".0%", title=""); fig.update_xaxes(dtick=1, title="")
        st.plotly_chart(style(fig), width="stretch")
    with t2:
        rb = an.segment_table(df, "risk_band"); rb["risk_band"] = pd.Categorical(rb.risk_band, an.RISK_ORDER, ordered=True); rb = rb.sort_values("risk_band")
        fig = go.Figure(go.Bar(x=rb.risk_band.astype(str), y=rb.exposure / 1e6, marker_color=[RISK_COLORS[x] for x in rb.risk_band.astype(str)],
                               marker_cornerradius=4, customdata=np.c_[rb.default_rate, rb.accounts],
                               hovertemplate="%{x}<br>Exposure $%{y:,.0f}M<br>Actual default rate %{customdata[0]:.1%}<br>%{customdata[1]:,} accounts<extra></extra>"))
        fig.update_layout(title="Exposure by model risk band"); fig.update_yaxes(title="$M")
        st.plotly_chart(style(fig), width="stretch")

    with st.expander("Five portfolio insights (computed on the full portfolio)", expanded=False):
        for i, ins in enumerate(an.insights(df_all), 1):
            st.markdown(f"**{i}. {ins['title']}**")
            st.markdown(f"{ins['evidence']}  \n*So what:* {ins['so_what']}")

# ============================================================================= ACCOUNT VIEW
with tab_acc:
    st.caption("Pick an account (list is limited to the current filters, ordered by model PD) to see its score, key drivers and history.")
    top = df.sort_values("pd_lgbm", ascending=False)
    manual = st.text_input("Account id (optional - overrides the list)", placeholder="e.g. ACC200001")
    options = top.account_id.head(500).tolist()
    sel = manual.strip() if manual.strip() else st.selectbox("Account (top 500 by PD in view)", options)
    row = df_all[df_all.account_id == sel]
    if row.empty:
        st.error("Account not found."); st.stop()
    r = row.iloc[0]
    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("Model PD (LightGBM)", f"{r.pd_lgbm:.1%}", f"{r.pd_lgbm - df_all.pd_lgbm.mean():+.1%} vs avg", delta_color="inverse")
    a2.metric("Baseline PD (logistic)", f"{r.pd_lr:.1%}")
    a3.metric("Risk band", r.risk_band)
    a4.metric("Recorded outcome", r.default_label)
    a5.metric("Anomaly flag", "Flagged" if r.anomaly_flag == 1 else "No", f"score {r.anomaly_score:.2f}", delta_color="off")
    st.caption("PD is out-of-fold: this account was scored by a model that never saw its own label.")

    d1, d2 = st.columns([3, 2])
    with d1:
        sv = shap_df.loc[sel]
        feats = sv.abs().sort_values(ascending=False).head(8).index
        vals = [r[x] if x in r.index else np.nan for x in feats]
        lab = [f"{x} = {v:,.2f}" if isinstance(v, (int, float, np.floating, np.integer)) and v == v else f"{x} = {v}" for x, v in zip(feats, vals)]
        fig = go.Figure(go.Bar(y=lab[::-1], x=sv[feats].values[::-1], orientation="h", marker_color=[ORANGE if v > 0 else BLUE for v in sv[feats].values[::-1]],
                               hovertemplate="%{y}<br>Contribution to log-odds %{x:+.3f}<extra></extra>"))
        fig.update_layout(title="Key drivers (SHAP): orange raises risk, blue lowers it"); fig.update_xaxes(title="Contribution to default log-odds")
        st.plotly_chart(style(fig, 360), width="stretch")
    with d2:
        st.markdown("**Account & customer**")
        facts = {"Loan": f"{r.loan_type} - {r.loan_purpose}", "Amount": f"${r.loan_amount:,.0f}", "Rate / tenure": f"{r.interest_rate_pct if r.interest_rate_pct==r.interest_rate_pct else 'n/a'}% / {r.tenure_months} m",
                 "Originated": str(r.origination_date)[:10], "Channel": r.disbursal_channel, "Customer": f"{r.customer_id}, age {r.age}, {r.employment_type}",
                 "Income": f"${r.annual_income:,.0f} ({r.income_band})", "Region": f"{r.region}, {r.city_tier}", "Peer segment": r.peer_segment}
        st.table(pd.Series(facts, name="").to_frame())

    rep, bur, col = load_account_detail(sel, r.customer_id)
    h1, h2 = st.columns(2)
    with h1:
        fig = go.Figure()
        fig.add_bar(x=rep.payment_month, y=rep.days_past_due, name="Days past due", marker_color=ORANGE, marker_cornerradius=3)
        fig.update_layout(title="Repayment: days past due by month since origination"); fig.update_xaxes(dtick=1, title="Month"); fig.update_yaxes(title="DPD")
        st.plotly_chart(style(fig, 280), width="stretch")
    with h2:
        pr = (rep.emi_paid / rep.emi_due).clip(upper=1.2)
        fig = go.Figure(go.Scatter(x=rep.payment_month, y=pr, mode="lines+markers", line=dict(color=BLUE, width=2), marker=dict(size=8), name="EMI paid / due"))
        fig.update_layout(title="Repayment: share of EMI paid"); fig.update_yaxes(tickformat=".0%", range=[0, 1.2]); fig.update_xaxes(dtick=1, title="Month")
        st.plotly_chart(style(fig, 280), width="stretch")
    bcol, ccol = st.columns(2)
    with bcol:
        st.markdown(f"**External bureau lines ({len(bur)})**")
        st.dataframe(bur.drop(columns=["bureau_id", "customer_id"]), hide_index=True, width="stretch", height=220)
    with ccol:
        st.markdown(f"**Collections actions ({len(col)})**")
        st.dataframe(col.drop(columns=["account_id"]), hide_index=True, width="stretch", height=220)
    if r.anomaly_flag == 1:
        st.warning("Flagged by anomaly detection: " + r.anomaly_reasons.replace(" || ", "  |  "))

# ============================================================================= ANOMALY REVIEW
with tab_anom:
    st.subheader("Anomaly detection: review queue")
    with st.expander("How this feature works - trigger, false positive, false negative, human review", expanded=True):
        st.markdown(
            "**What triggers a flag.** Each account is compared (1) with its *peer segment* (loan type x income band) on 11 repayment and bureau metrics "
            "using robust z-scores plus an IsolationForest, and (2) with its *own history* (last 3 months vs the first 9: DPD jump, sudden stop after a clean run, "
            "balance growing). The top 3% by anomaly score are flagged, and each flag lists its reasons.  \n"
            "**False positive:** an account that looks odd but is harmless, e.g. a one-off late payment on a large loan, or a customer whose profile is simply rare in its segment. "
            "Flagged accounts are not necessarily risky, they are *unusual*.  \n"
            "**False negative:** a genuinely deteriorating account that stays inside normal ranges (slow drift, a segment whose norm is already poor) or whose problem is in data we do not hold.  \n"
            "**Human in the loop:** the system never acts on a flag. An analyst opens the reasons and the account history, then records **Confirm** (needs follow-up) or **Dismiss** (explained/benign) with a note. "
            "Decisions are logged with reviewer and time; dismissal rates per reason should be used to retune thresholds.")
    fl = df[df.anomaly_flag == 1].copy()
    fl["status"] = fl.account_id.map(last_dec.decision).fillna("Unreviewed")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Flagged in view", f"{len(fl):,}"); m2.metric("Unreviewed", f"{(fl.status=='Unreviewed').sum():,}")
    m3.metric("Confirmed", f"{(fl.status=='Confirmed').sum():,}"); m4.metric("Dismissed", f"{(fl.status=='Dismissed').sum():,}")
    q1, q2 = st.columns(2)
    lens_f = q1.multiselect("Detection lens", sorted(fl.anomaly_lens.unique()), default=[])
    stat_f = q2.multiselect("Status", ["Unreviewed", "Confirmed", "Dismissed"], default=["Unreviewed"])
    if lens_f: fl = fl[fl.anomaly_lens.isin(lens_f)]
    if stat_f: fl = fl[fl.status.isin(stat_f)]
    fl = fl.sort_values("anomaly_score", ascending=False)
    queue = fl[["account_id", "loan_type", "anomaly_score", "anomaly_lens", "pd_lgbm", "status", "anomaly_reasons"]].copy()
    queue["anomaly_reasons"] = queue.anomaly_reasons.str.replace(" || ", "  |  ", regex=False)
    st.dataframe(queue.rename(columns={"anomaly_reasons": "why flagged"}), hide_index=True, width="stretch", height=300,
                 column_config={"pd_lgbm": st.column_config.NumberColumn("Model PD", format="%.2f"), "anomaly_score": st.column_config.NumberColumn("Score", format="%.3f"),
                                "why flagged": st.column_config.TextColumn(width="large")})
    if len(fl):
        pick = st.selectbox("Open a flagged account to review", fl.account_id.tolist(), key="anom_pick")
        pr_ = df_all[df_all.account_id == pick].iloc[0]
        st.markdown("**Why it was flagged**")
        for rs in pr_.anomaly_reasons.split(" || "):
            st.write("-", rs)
        rep, _, _ = load_account_detail(pick, pr_.customer_id)
        fig = go.Figure(go.Bar(x=rep.payment_month, y=rep.days_past_due, marker_color=[ORANGE if m >= 10 else GREY for m in rep.payment_month], marker_cornerradius=3,
                               hovertemplate="Month %{x}<br>DPD %{y}<extra></extra>"))
        fig.update_layout(title="Days past due by month (last 3 months highlighted)"); fig.update_xaxes(dtick=1)
        st.plotly_chart(style(fig, 240), width="stretch")
        prev = last_dec.loc[pick] if pick in last_dec.index else None
        if prev is not None:
            st.info(f"Last decision: {prev.decision} by {prev.reviewer} at {prev.decided_at}. Note: {prev.note if isinstance(prev.note, str) else '-'}")
        # Displaying a reviewer name does not authenticate the reviewer.
        import hmac
        import os
        try:
            review_token = st.secrets.get("REVIEW_WRITE_TOKEN", "")
        except Exception:
            review_token = os.environ.get("REVIEW_WRITE_TOKEN", "")
        supplied_token = st.text_input("Review authorization token", type="password", key="review_auth") if review_token else ""
        can_review = bool(review_token) and hmac.compare_digest(str(supplied_token), str(review_token))
        if not review_token:
            st.info("Decision submissions are disabled until REVIEW_WRITE_TOKEN is configured.")
        elif not can_review:
            st.info("Enter the review token before submitting a decision.")
        with st.form("decision", clear_on_submit=True):
            reviewer = st.text_input("Reviewer name", value="analyst")
            note = st.text_input("Note (optional)")
            b1, b2 = st.columns(2)
            ok = b1.form_submit_button("Confirm - needs follow-up", type="primary")
            no = b2.form_submit_button("Dismiss - benign / explained")
            if (ok or no) and not can_review:
                st.error("Review authorization is required.")
            elif ok or no:
                rec = pd.DataFrame([{"account_id": pick, "decision": "Confirmed" if ok else "Dismissed", "note": note, "reviewer": reviewer or "analyst",
                                     "decided_at": datetime.now().isoformat(timespec="seconds")}])
                DECISIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
                rec.to_csv(DECISIONS_FILE, mode="a", header=not DECISIONS_FILE.exists(), index=False)
                st.success(f"Recorded: {rec.decision.iloc[0]}"); st.rerun()
        st.caption("Decisions are saved to data/processed/anomaly_decisions.csv. On free hosting the disk is ephemeral; production would write to a database.")
    else:
        st.info("No flagged accounts match.")

# ============================================================================= ASK THE PORTFOLIO
with tab_ask:
    st.subheader("Ask the Portfolio")
    with st.expander("How this feature works - trigger, false positive, false negative, human review", expanded=True):
        st.markdown(
            "**What triggers it.** Each question you submit. The LLM writes a read-only SQL query; the app runs it on the dashboard data (an in-memory copy, SELECT only, row-capped) "
            "and the LLM answers *only* from the returned rows. The SQL and result rows are shown under every answer, and every number in the answer is checked against those rows.  \n"
            "**Data sent to the LLM:** the table schema, your question, and the aggregated rows returned by the query (max 60). No raw files, names or contact details exist in the data.  \n"
            "**False positive:** a confident answer that is wrong, e.g. the SQL filters on the wrong column or the question was ambiguous ('exposure' means original loan amount here). "
            "The number check catches invented figures but not a mis-specified query - read the SQL.  \n"
            "**False negative:** a question the data *can* answer but the assistant declines or fails to query, in which case rephrase or use the other tabs.  \n"
            "**Human in the loop:** the analyst reads the SQL and rows before relying on an answer; questions about forecasts, causes, outside facts or individuals are declined by design.")
    key = llm.get_api_key()
    if key:
        st.success(f"LLM mode: {llm.MODEL} with grounded SQL tool.")
    else:
        st.warning("No ANTHROPIC_API_KEY is set, so a built-in set of question patterns is used (same SQL path, same grounding). Add a key for free-form questions.")

    @st.cache_resource(show_spinner=False)
    def get_con():
        return llm.build_db(df_all, metrics, decisions)

    st.session_state.setdefault("qa_history", [])
    examples = ["Which loan type has the highest default rate?", "What is the default rate by income band?", "How much exposure sits in Mortgages in the South region?",
                "How many flagged accounts are still unreviewed?", "What is the model's Gini on the hold-out set?", "Will defaults rise next year?"]
    ex = st.columns(3)
    for i, e in enumerate(examples):
        if ex[i % 3].button(e, key=f"ex{i}", width="stretch"):
            st.session_state["q"] = e
    q = st.text_input("Your question", key="q", placeholder="e.g. Which region has the lowest default rate among Auto Loans?")
    if st.button("Ask", type="primary") and q.strip():
        with st.spinner("Querying the portfolio..."):
            con = llm.build_db(df_all, metrics, an.load_decisions())
            st.session_state.qa_history.insert(0, (q, llm.ask(q, con)))
    for qq, a in st.session_state.qa_history[:5]:
        with st.container(border=True):
            st.write("Q:", qq)
            if a.status == "declined":
                st.info(f"Declined: {a.text}")
            elif a.status == "error":
                st.error(a.text)
            else:
                st.write(a.text)
                if a.unverified:
                    st.warning("These figures could not be traced to the query results: " + ", ".join(a.unverified) + ". Do not rely on them.")
                else:
                    st.caption("All figures in this answer match the query results below.")
            st.caption(f"Engine: {'LLM + SQL tool' if a.engine == 'llm' else 'built-in patterns (no LLM)'}")
            for ev in a.evidence:
                with st.expander("Source: SQL and result rows"):
                    st.code(ev["sql"], language="sql"); st.dataframe(ev["rows"], hide_index=True, width="stretch")

# ============================================================================= MODEL
with tab_model:
    st.subheader("Default model")
    h = metrics["headline"]
    cols_ = st.columns(4)
    cols_[0].metric("LightGBM AUC (hold-out)", f"{h['lightgbm']['auc']:.3f}"); cols_[1].metric("Gini", f"{h['lightgbm']['gini']:.3f}")
    cols_[2].metric("KS", f"{h['lightgbm']['ks']:.3f}"); cols_[3].metric("Logistic baseline AUC", f"{h['logistic_regression']['auc']:.3f}")
    st.caption(f"Hold-out = 30% of customers ({metrics['n_test']:,} accounts), grouped by customer so nobody appears in both train and test. "
               "Discrimination is modest: the synthetic data carries a moderate, noisy signal, and a heavily regularised shallow boosted model is only on par with logistic regression.")
    rows = []
    for name, v in metrics["feature_sets"].items():
        for mdl, lab in [("logistic_regression", "Logistic regression"), ("lightgbm", "LightGBM")]:
            rows.append({"Feature set": name, "Model": lab, **{k.upper(): x for k, x in v[mdl].items()}})
    rows += [{"Feature set": "C. full, out-of-time (train <=2021, test 2022-23)", "Model": lab, **{k.upper(): x for k, x in metrics["out_of_time"][m].items()}}
             for m, lab in [("logistic_regression", "Logistic regression"), ("lightgbm", "LightGBM")]]
    rows += [{"Feature set": "C. full, 5-fold out-of-fold", "Model": lab, **{k.upper(): x for k, x in metrics["oof"][m].items()}}
             for m, lab in [("logistic_regression", "Logistic regression"), ("lightgbm", "LightGBM")]]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    m1, m2 = st.columns(2)
    with m1:
        tf = pd.DataFrame(metrics["top_features"]).sort_values("mean_abs_shap")
        fig = go.Figure(go.Bar(y=tf.feature, x=tf.mean_abs_shap, orientation="h", marker_color=BLUE, marker_cornerradius=3))
        fig.update_layout(title="Global feature importance (mean |SHAP|, LightGBM)")
        st.plotly_chart(style(fig, 420), width="stretch")
    with m2:
        cal = pd.DataFrame(metrics["calibration_deciles"])
        fig = go.Figure()
        fig.add_scatter(x=cal.mean_pred, y=cal.actual_rate, mode="lines+markers", name="Deciles", line=dict(color=BLUE, width=2), marker=dict(size=8))
        fig.add_scatter(x=[0, .5], y=[0, .5], mode="lines", name="Perfect calibration", line=dict(color=GREY, dash="dot"))
        fig.update_layout(title="Calibration by score decile (out-of-fold)"); fig.update_xaxes(title="Predicted PD", tickformat=".0%"); fig.update_yaxes(title="Actual default rate", tickformat=".0%")
        st.plotly_chart(style(fig, 420), width="stretch")
    st.markdown("**Sensitivity checks (LightGBM hold-out AUC)**")
    st.dataframe(pd.Series(metrics["sensitivity_auc_lightgbm"], name="AUC").to_frame(), width="stretch")
    st.markdown("**Monitoring: PSI, development (<=2021 originations) vs recent (2022-23)**  (<0.10 stable, 0.10-0.25 watch, >0.25 act)")
    st.dataframe(pd.DataFrame(metrics["monitoring_psi_dev_vs_recent"]), hide_index=True, width="stretch")
    st.markdown(
        "**How the model would be monitored after deployment**\n"
        "- *Stability, monthly:* PSI on the score and the top features against the development sample; investigate above 0.10.\n"
        "- *Performance, quarterly (once outcomes mature):* AUC/KS/Gini by vintage and by loan type, calibration by decile; retrain or recalibrate if AUC falls more than 0.03 or any decile is off by more than 20% relative.\n"
        "- *Data quality, daily:* null/zero rates and range checks on bureau and repayment feeds; the thin-file share.\n"
        "- *Fairness and policy:* approval and default rates by age band and region; gender and marital status are not model inputs.\n"
        "- *Governance:* champion/challenger (logistic regression is the standing challenger), versioned features, and a documented override process.")

# ============================================================================= DQ
with tab_dq:
    st.subheader("Data-quality log and join logic")
    st.markdown("**Join logic.** `loan_accounts` is the spine (one row per account, 14,685 rows, never reduced). LEFT joins: customers on `customer_id` (m:1), bureau lines "
                "aggregated per customer then joined on `customer_id` (m:1), repayment rows aggregated per account on `account_id` (1:1), collections aggregated per account (1:1). "
                "Row count is asserted unchanged after every join.")
    st.dataframe(dq, hide_index=True, width="stretch", column_config={"decision": st.column_config.TextColumn(width="large"), "issue": st.column_config.TextColumn(width="large")})