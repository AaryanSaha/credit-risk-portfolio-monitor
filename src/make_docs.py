"""Builds docs/architecture.pdf and docs/solution_overview.pdf (one page each) from the computed results.  python -m src.make_docs"""
import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import Frame, KeepInFrame, Paragraph

from .analytics import insights, load_portfolio
from .config import PROCESSED, ROOT

BLUE, ORANGE, AQUA, INK, MUTE, LINE = colors.HexColor("#2a78d6"), colors.HexColor("#eb6834"), colors.HexColor("#1baf7a"), colors.HexColor("#1f1f1d"), colors.HexColor("#52514e"), colors.HexColor("#c9c8c2")
LIGHT = colors.HexColor("#eef4fc"); LORANGE = colors.HexColor("#fdf0ea"); LAQUA = colors.HexColor("#e7f6f0")
DOCS = ROOT / "docs"; DOCS.mkdir(exist_ok=True)


def box(c, x, y, w, h, title, lines, fill=LIGHT, edge=BLUE, tsize=8.5, lsize=6.8):
    c.setFillColor(fill); c.setStrokeColor(edge); c.setLineWidth(1)
    c.roundRect(x, y, w, h, 4, fill=1, stroke=1)
    c.setFillColor(INK); c.setFont("Helvetica-Bold", tsize); c.drawString(x + 5, y + h - 11, title)
    st = ParagraphStyle("b", fontName="Helvetica", fontSize=lsize, leading=lsize + 1.8, textColor=MUTE)
    f = Frame(x + 2, y + 1, w - 4, h - 14, leftPadding=3, rightPadding=3, topPadding=0, bottomPadding=0, showBoundary=0)
    f.addFromList([KeepInFrame(w - 10, h - 15, [Paragraph(l, st) for l in lines], mode="shrink")], c)


def arrow(c, x1, y1, x2, y2, color=INK, label=None):
    c.setStrokeColor(color); c.setFillColor(color); c.setLineWidth(1.2); c.line(x1, y1, x2, y2)
    import math
    a = math.atan2(y2 - y1, x2 - x1); s = 5
    p = c.beginPath(); p.moveTo(x2, y2); p.lineTo(x2 - s * math.cos(a - .4), y2 - s * math.sin(a - .4)); p.lineTo(x2 - s * math.cos(a + .4), y2 - s * math.sin(a + .4)); p.close(); c.drawPath(p, fill=1, stroke=0)
    if label:
        c.setFont("Helvetica-Oblique", 6.3); c.setFillColor(MUTE); c.drawCentredString((x1 + x2) / 2, (y1 + y2) / 2 + 3, label)


def architecture():
    W, H = landscape(A4)
    c = canvas.Canvas(str(DOCS / "architecture.pdf"), pagesize=(W, H)); c.setTitle("Architecture - Credit Risk Portfolio Monitor")
    c.setFillColor(INK); c.setFont("Helvetica-Bold", 14); c.drawString(15 * mm, H - 14 * mm, "Architecture: raw data to dashboard, with both AI features")
    c.setFont("Helvetica", 7.5); c.setFillColor(MUTE)
    c.drawString(15 * mm, H - 19 * mm, "Everything runs as one Python 3.13 app. Blue = batch pipeline (python -m src.build). Orange = AI features. Green = human decisions.")
    bw, bh, y = 46 * mm, 62 * mm, H - 94 * mm
    xs = [15 * mm + i * (bw + 9 * mm) for i in range(5)]
    box(c, xs[0], y, bw, bh, "1. Raw data  (data/raw)", ["5 CSVs, synthetic bank data:", "customers 10,000", "loan_accounts 14,685 (target)", "bureau_credit 22,979", "repayment_history 176,220", "collections_events 8,635", "<b>Tool:</b> pandas read_csv"])
    box(c, xs[1], y, bw, bh, "2. Ingestion  (src/ingest.py)", ["Clean: gender labels, missing values (flag + impute), implausible tenure, bureau status/date conflicts.", "LEFT-join from loan_accounts; row count asserted.", "Aggregate bureau (per customer), repayment, collections (per account).", "Logs every decision to dq_log.csv.", "<b>Tool:</b> pandas, pyarrow"])
    box(c, xs[2], y, bw, bh, "3. Analytics  (src/analytics.py)", ["Default rate, exposure, share of defaulted exposure by income, employment, region, age, loan type.", "Cross-dimension cuts and 5 computed insights.", "<b>Tool:</b> pandas, Wilson intervals"])
    box(c, xs[3], y, bw, bh, "4. Model  (src/model.py)", ["Logistic regression baseline + LightGBM.", "Customer-grouped splits; 5-fold out-of-fold scores.", "AUC / KS / Gini, SHAP drivers, PSI and calibration for monitoring.", "Gender, marital status, collections excluded.", "<b>Tool:</b> scikit-learn, LightGBM, SHAP"])
    box(c, xs[4], y, bw, bh, "5. Data products", ["<i>data/processed</i>", "analysis_table.parquet (1 row/account)", "scores.parquet (PD, band)", "shap_values.parquet", "anomalies.parquet", "model_metrics.json", "dq_log.csv", "anomaly_decisions.csv (analyst log)"], fill=colors.HexColor("#f4f4f2"), edge=MUTE)
    for i in range(4):
        arrow(c, xs[i] + bw, y + bh / 2, xs[i + 1], y + bh / 2)
    # dashboard
    dy = y - 52 * mm
    c.setFillColor(colors.HexColor("#fafaf9")); c.setStrokeColor(INK); c.setLineWidth(1.2); c.roundRect(15 * mm, dy - 6 * mm, W - 30 * mm, 46 * mm, 5, fill=1, stroke=1)
    c.setFillColor(INK); c.setFont("Helvetica-Bold", 9); c.drawString(19 * mm, dy + 34 * mm, "6. Dashboard  (app/app.py, Streamlit; hosted on Streamlit Community Cloud, API key in its secrets store)")
    tw = (W - 30 * mm - 6 * mm * 2 - 8 * mm) / 3
    box(c, 19 * mm, dy - 2 * mm, tw, 30 * mm, "Portfolio + Account views", ["Sidebar filters (loan type, region, employment, income, age, year) drive KPIs, segment bars, heatmap cross-cut, trends, risk bands.", "Account view: PD, SHAP key drivers, 12-month repayment, bureau lines, collections.", "<b>Tool:</b> Plotly"], fill=LIGHT, edge=BLUE)
    box(c, 19 * mm + tw + 6 * mm, dy - 2 * mm, tw, 30 * mm, "AI feature 2: Anomaly review", ["Peer-segment robust z + IsolationForest, plus own-history (last 3 vs first 9 months). Top 3% flagged with plain-English reasons.", "<b>No LLM.</b> Analyst clicks Confirm / Dismiss + note."], fill=LORANGE, edge=ORANGE)
    box(c, 19 * mm + 2 * (tw + 6 * mm), dy - 2 * mm, tw, 30 * mm, "AI feature 1: Ask the Portfolio", ["Question -> LLM writes read-only SQL -> app runs it -> LLM answers from the rows only. SQL + rows shown; numbers re-checked. Unanswerable questions declined.", "No key: built-in question patterns, same SQL path."], fill=LORANGE, edge=ORANGE)
    for x in [xs[4] + bw / 2]:
        arrow(c, x, y, x, dy + 40 * mm - 6 * mm, label="read")
    # LLM box
    ly = dy - 51 * mm
    lx = 19 * mm + 2 * (tw + 6 * mm)
    box(c, lx, ly, tw, 40 * mm, "LLM API call (src/llm.py)",
        ["Anthropic Messages API with tool use. <b>Sent:</b> system rules, table schema + column descriptions, the user's question, and the rows returned by each SQL (max 60 rows, aggregates by default).",
         "<b>Not sent:</b> raw CSVs, the full table, API key. The data holds no names or contact details, only pseudonymous ids.",
         "<b>Guards:</b> SELECT-only on an in-memory SQLite copy, 5 s timeout, 60-row cap, one statement; number check on the final answer."], fill=LORANGE, edge=ORANGE)
    arrow(c, lx + tw / 2, dy - 2 * mm, lx + tw / 2, ly + 40 * mm, ORANGE, "question + schema / rows")
    box(c, 15 * mm, ly, 85 * mm, 40 * mm, "Human-in-the-loop & audit", ["Anomaly flags are only suggestions: the analyst records Confirm or Dismiss, reviewer, note and time (anomaly_decisions.csv; a database in production).", "Ask the Portfolio shows its SQL so the analyst can check the logic before using a number.", "Model scores inform, never auto-decline."], fill=LAQUA, edge=AQUA)
    box(c, 104 * mm, ly, 87 * mm, 40 * mm, "Delivery & ops", ["GitHub repo (no secrets) -> Streamlit Community Cloud; the app builds data/model on first start if missing.", "Secrets: ANTHROPIC_API_KEY via st.secrets / env; absent key = degraded but working fallback.", "tests/ covers joins, leakage guard, SQL guard, grounding check, LLM tool loop (mocked)."], fill=colors.HexColor("#f4f4f2"), edge=MUTE)
    c.showPage(); c.save()


def overview():
    m = json.load(open(PROCESSED / "model_metrics.json")); df = load_portfolio(); ins = insights(df)
    W, H = A4
    c = canvas.Canvas(str(DOCS / "solution_overview.pdf"), pagesize=A4); c.setTitle("Solution overview - Credit Risk Portfolio Monitor")
    h1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=9.4, leading=11.5, textColor=BLUE, spaceBefore=6, spaceAfter=1)
    p = ParagraphStyle("p", fontName="Helvetica", fontSize=7.9, leading=10, textColor=INK, spaceAfter=2.5)
    c.setFillColor(INK); c.setFont("Helvetica-Bold", 13); c.drawString(14 * mm, H - 13 * mm, "Credit Risk Portfolio Monitor - solution overview")
    c.setFont("Helvetica", 7); c.setFillColor(MUTE); c.drawString(14 * mm, H - 17.5 * mm, "Streamlit dashboard + grounded LLM Q&A + anomaly review, on 10,000 customers / 14,685 loan accounts / 176k monthly repayment rows (synthetic).")
    h = m["headline"]; oot = m["out_of_time"]; o = m["oof"]
    tot = df.loan_amount.sum(); fl = df[df.anomaly_flag == 1]
    left = [
        ("What the dashboard shows and how an analyst uses it", [
            "<b>Portfolio:</b> KPIs (accounts, exposure $%.0fM, default rate %.1f%%), default rate by any of 11 dimensions with share of defaulted exposure, a two-dimension heatmap, vintage trends and exposure by model risk band. Sidebar filters re-cut every view; the analyst starts here to find where losses concentrate." % (tot / 1e6, df.target_default.mean() * 100),
            "<b>Account view:</b> pick an account to see model PD, SHAP key drivers, 12-month repayment, external bureau lines and collections; used to decide what to do with a segment's worst names.",
            "<b>Anomaly review:</b> queue of flagged accounts with reasons; analyst confirms or dismisses. <b>Ask the Portfolio:</b> free-text questions with SQL and rows as proof. <b>Model &amp; monitoring</b> and <b>Data quality</b> tabs document performance and every cleaning decision."]),
        ("AI feature 1 - Ask the Portfolio (LLM)", [
            "<b>How:</b> LLM writes a read-only SQL query, the app executes it on an in-memory copy of the dashboard data, the LLM answers only from the returned rows; SQL, rows and a number-check result are shown. <b>Trigger:</b> each submitted question. <b>Sent to LLM:</b> schema, question, at most 60 result rows.",
            "<b>False positive:</b> confident answer from a mis-specified query (wrong column, ambiguous 'exposure'); the number check catches invented figures, not wrong logic. <b>False negative:</b> answerable question declined or not queried. <b>Human step:</b> analyst reads the SQL before relying on the figure; forecasts, causes and outside facts are declined."]),
        ("AI feature 2 - Anomaly detection", [
            "<b>How:</b> 11 repayment and bureau metrics as robust z-scores against the peer segment (loan type x income band) + IsolationForest; own-history checks compare months 10-12 with months 1-9 (DPD jump, sudden stop after clean run, balance growing). Top 3%% flagged (%d accounts: %d own-history, %d peer, %d both). <b>Trigger:</b> score above the 97th percentile." % (len(fl), (fl.anomaly_lens == "Own history").sum(), (fl.anomaly_lens == "Peer segment").sum(), (fl.anomaly_lens == "Peer + Own history").sum()),
            "<b>False positive:</b> unusual but harmless (one late payment on a big loan, rare profile). <b>False negative:</b> slow drift or a segment whose norm is already poor. <b>Human step:</b> analyst records Confirm / Dismiss + note; dismissal rates per reason retune thresholds. Sanity check: flagged accounts default at %.1f%% vs %.1f%% for the rest (a validation, not the objective)." % (fl.target_default.mean() * 100, df[df.anomaly_flag == 0].target_default.mean() * 100)]),
        ("Model results and approach", [
            "Target = target_default (%.1f%% base rate). Features: application, loan, 15 bureau aggregates, 11 repayment aggregates. Collections, gender and marital status excluded (collections follow delinquency and added no lift: AUC +0.0005; protected attributes). Customer-grouped 70/30 split." % (df.target_default.mean() * 100),
            "<b>Hold-out (%s accounts):</b> LightGBM AUC %.3f, Gini %.3f, KS %.3f; logistic baseline AUC %.3f, Gini %.3f, KS %.3f. <b>Out-of-time</b> (train to 2021, test 2022-23): LightGBM AUC %.3f; 5-fold out-of-fold AUC %.3f. Boosting is tuned shallow (3 leaves) and is only on par with logistic regression: the honest reading is moderate signal, so both are kept." % (f"{m['n_test']:,}", h["lightgbm"]["auc"], h["lightgbm"]["gini"], h["lightgbm"]["ks"], h["logistic_regression"]["auc"], h["logistic_regression"]["gini"], h["logistic_regression"]["ks"], oot["lightgbm"]["auc"], o["lightgbm"]["auc"]),
            "Top SHAP drivers: log income, on-time payment share, years employed, external mean DPD, age. <b>Monitoring:</b> monthly PSI on score and top features (all < 0.01 dev vs recent), quarterly AUC/KS and decile calibration by vintage, daily feed checks, champion/challenger."]),
    ]
    right = [
        ("Top 3 portfolio insights", [f"<b>{i+1}. {x['title']}.</b> {x['evidence']} <i>So what:</i> {x['so_what']}" for i, x in enumerate(ins[:3])]),
        ("Key assumptions", [
            "Exposure is the original loan amount (no current balance). Bureau data is treated as a snapshot: 68%% of lines opened after the customer's first loan, so a point-in-time check was run (AUC %.3f vs %.3f) and production needs as-of-origination pulls. 12 repayment months per account regardless of tenure. Collections dates are unreliable (45%% pre-date origination) so only counts are used. Missing values are flagged and imputed, never dropped. Synthetic data: results show method, not real-world loss." % (m["sensitivity_auc_lightgbm"]["point-in-time bureau only"], m["sensitivity_auc_lightgbm"]["baseline (full feature set)"])]),
        ("What production needs from the bank's IT and data teams", [
            "<b>Data:</b> as-of-date bureau pulls at application; monthly repayment feed with calendar dates; current balances and collateral values (LTV); default definition (90+ DPD?) and write-off dates; collections timestamps. <b>Platform:</b> read-only warehouse access or a nightly extract; SSO and role-based access; a database for analyst decisions with audit trail; secrets management and an approved LLM endpoint (data-residency, no training on inputs, logging).",
            "<b>Governance:</b> model risk sign-off, fairness review, a policy for LLM-generated answers (SQL always visible), and a feedback loop so confirmed anomalies become labelled data."]),
    ]
    def col(items, x):
        story = []
        for t, ps in items:
            story.append(Paragraph(t, h1)); story += [Paragraph(s, p) for s in ps]
        f = Frame(x, 10 * mm, (W - 28 * mm - 6 * mm) / 2, H - 33 * mm, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        kif = KeepInFrame((W - 28 * mm - 6 * mm) / 2, H - 33 * mm, story, mode="shrink")
        f.addFromList([kif], c)
    cw = (W - 28 * mm - 6 * mm) / 2
    col(left, 14 * mm); col(right, 14 * mm + cw + 6 * mm)
    c.setStrokeColor(LINE); c.line(14 * mm + cw + 3 * mm, 10 * mm, 14 * mm + cw + 3 * mm, H - 24 * mm)
    c.showPage(); c.save()


if __name__ == "__main__":
    architecture(); overview(); print("docs written")