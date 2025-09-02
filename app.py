"""
MDF Allocation Recommender — Partner-Fairness Layer + Human-Friendly Tables
----------------------------------------------------------------------------
Run:
  - Keep your CSVs under ./data (as you already have them).
  - Optional: set GOOGLE_API_KEY for the LLM.
  - python app.py

Outputs:
  - ./output/plot.png
  - ./output/recommendations.csv  (campaign-type allocation)
  - ./output/report.pdf           (includes BOTH tables + plot + expanded narrative)
"""

import os
import math
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle, SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

try:
    import google.generativeai as genai
except Exception:
    genai = None

# Default LLM model
LLM_MODEL = "gemma-3-27b-it"

# Data folder
DATA_DIR = "data"

# MDF parameters
TOTAL_BUDGET = 15000
EXPLORE_FRAC = 0.25
CLOSE_RATE = 0.10

# Output patameters
OUTPUT_DIR = "output"
OUTPUT_PLOT = os.path.join(OUTPUT_DIR, "plot.png")
OUTPUT_RECO = os.path.join(OUTPUT_DIR, "recommendations.csv")
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "report.pdf")

# "mdf_outcomes_sample.csv" contains GRANTS (amount).
# "mdf_grants_sample.csv"   contains OUTCOMES (pipeline, etc.).
FILE_GRANTS   = os.path.join(DATA_DIR, "mdf_outcomes_sample.csv")       # grants (amount)
FILE_OUTCOMES = os.path.join(DATA_DIR, "mdf_grants_sample.csv")         # outcomes (pipeline, closed_won, tier, cap)
FILE_BASELINE = os.path.join(DATA_DIR, "baseline_pipeline_next3.csv")   # base pipeline

# How many partners to split fairly across (easy to change: 2 or 3).
TOP_PARTNER_COUNT = 3  # set to 2 if you want a top-2 fairness split

# --- Utilities -----------------------------------------------------------------
def ensure_files_exist():
    for f in [FILE_GRANTS, FILE_OUTCOMES, FILE_BASELINE]:
        if not os.path.exists(f):
            raise FileNotFoundError(
                f"Data file not found: {f}. Ensure sample files are in the 'data' folder."
            )

def coerce_month(month_col):
    """Coerce to the first day of the month (Timestamp)."""
    return pd.to_datetime(month_col).dt.to_period("M").dt.to_timestamp()

def _load_and_map(path, required_cols, col_map):
    """
    Helper to load a CSV and map columns to a required schema.
    Keeps your original, minimal approach.
    """
    df = pd.read_csv(path)
    df = df.rename(columns=col_map)
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(
            f"CSV '{os.path.basename(path)}' missing required column(s): {missing}. "
            f"Columns found: {list(df.columns)}"
        )
    return df[list(required_cols)]

def load_grants_csv(path):
    """
    GRANTS schema (as you used it):
    month | partner | campaign_type | amount
    """
    req = set(["month", "partner", "campaign_type", "amount"])
    col_map = {"campaign": "campaign_type", "grant": "amount", "spend": "amount"}
    return _load_and_map(path, req, col_map)

def load_outcomes_csv(path):
    """
    OUTCOMES schema:
    month | partner | pipeline | closed_won | tier | cap_monthly
    """
    req = set(["month", "partner", "pipeline", "closed_won", "tier", "cap_monthly"])
    col_map = {"pipeline_created": "pipeline", "tier_level": "tier"}
    return _load_and_map(path, req, col_map)

def load_baseline_csv(path):
    """Baseline schema: month | partner | baseline_pipeline"""
    req = set(["month", "partner", "baseline_pipeline"])
    col_map = {"baseline": "baseline_pipeline"}
    return _load_and_map(path, req, col_map)


# --- Core (kept as in your app, with your logic) -------------------------------
def estimate_lag(grants_df, outcomes_df, lags=(2, 3)):
    """
    Estimate lag (months) that best aligns grants -> pipeline.
    Your original correlation-based approach is preserved.
    """
    best_corr = -1
    best_lag = lags[0]
    for lag in lags:
        grants_df["outcome_month"] = grants_df["month"] + pd.DateOffset(months=lag)
        merged = pd.merge(
            grants_df,
            outcomes_df,
            left_on=["outcome_month", "partner"],
            right_on=["month", "partner"],
            suffixes=("_grant", "_outcome"),
        )
        corr = merged["amount"].corr(merged["pipeline"])
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    return best_lag

def roi_posteriors(grants_df, outcomes_df, lag, prior_alpha=1.0, prior_beta=10000.0):
    """
    Simple Gamma posterior per campaign_type using your structure:
    post_alpha = prior_alpha + total_pipeline
    post_beta  = prior_beta  + total_amount
    """
    grants_df["outcome_month"] = grants_df["month"] + pd.DateOffset(months=lag)
    merged = pd.merge(
        grants_df,
        outcomes_df,
        left_on=["outcome_month", "partner"],
        right_on=["month", "partner"],
        suffixes=("_grant", "_outcome"),
    )
    agg = merged.groupby("campaign_type").agg(
        total_amount=("amount", "sum"), total_pipeline=("pipeline", "sum"), n=("amount", "count")
    )
    agg["post_alpha"] = prior_alpha + agg["total_pipeline"]
    agg["post_beta"]  = prior_beta  + agg["total_amount"]
    agg["n_obs"] = agg["n"]
    return agg[["post_alpha", "post_beta", "n_obs"]].reset_index()

def thompson_allocate(posteriors_df, total_budget, explore_frac=0.25, seed=None):
    """
    Thompson sampling across campaign types (your original idea).
    Returns columns: campaign_type, prob_best, allocation, uplift
    """
    rng = np.random.default_rng(seed)
    reco = posteriors_df.copy()
    reco["roi_samples"] = [
        rng.gamma(row.post_alpha, 1 / row.post_beta, size=1000) for row in reco.itertuples()
    ]
    roi_matrix = np.array(reco["roi_samples"].tolist()).T
    best_arm_indices = np.argmax(roi_matrix, axis=1)
    counts = np.bincount(best_arm_indices, minlength=len(reco))
    reco["prob_best"] = counts / len(best_arm_indices)

    obs = reco[reco["n_obs"] > 0].copy()
    unobs = reco[reco["n_obs"] == 0].copy()

    exploit_budget = total_budget * (1 - explore_frac)
    obs["allocation"] = obs["prob_best"] * exploit_budget

    explore_budget = total_budget * explore_frac
    if not unobs.empty:
        unobs["allocation"] = explore_budget / len(unobs)
    else:
        obs["allocation"] += obs["prob_best"] * explore_budget

    final_reco = pd.concat([obs, unobs])
    final_reco["uplift"] = final_reco.apply(
        lambda r: rng.gamma(r.post_alpha, 1 / r.post_beta) * r.allocation, axis=1
    )

    final_reco["allocation"] = final_reco["allocation"].round(0)
    final_reco["uplift"]     = final_reco["uplift"].round(0)

    return final_reco[["campaign_type", "prob_best", "allocation", "uplift"]]

def apply_uplift_to_baseline(baseline_next3, current_month, lag, reco, close_rate):
    """
    Keep your earlier approach: add campaign-level uplift to the baseline at impact month.
    """
    impact_month = current_month + pd.DateOffset(months=lag)
    total_uplift_pipeline = reco["uplift"].sum()
    total_uplift_won = total_uplift_pipeline * close_rate  # kept for completeness (not used explicitly below)

    baseline_for_month = baseline_next3[baseline_next3["month"] == impact_month]["baseline_pipeline"].sum()
    with_mdf = baseline_next3.groupby("month")["baseline_pipeline"].sum()
    with_mdf.loc[impact_month] = baseline_for_month + total_uplift_pipeline
    with_mdf = with_mdf.sort_index()
    return with_mdf, total_uplift_pipeline, impact_month

def make_impact_plot(baseline_df, with_mdf_series, out_path):
    """Plot baseline vs with-MDF totals (simple & clear)."""
    baseline_series = baseline_df.groupby("month")["baseline_pipeline"].sum()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(baseline_series.index, baseline_series.values, label="Baseline Pipeline",
            marker="o", linestyle="--", color="gray")
    ax.plot(with_mdf_series.index, with_mdf_series.values, label="Forecast with MDF",
            marker="o", linestyle="-", color="royalblue")

    ax.set_title("MDF Impact on Pipeline Forecast", fontsize=16, pad=20)
    ax.set_ylabel("Pipeline ($)", fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Impact plot saved to: {out_path}")


# --- Partner-Fairness Layer ----------------------------------------------
def partner_topk_fair_allocation(
    grants_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    lag: int,
    total_budget: float,
    top_k: int = TOP_PARTNER_COUNT,
) -> pd.DataFrame:
    """
    Goal:
      Choose the top-K partners by historical ROI (pipeline / amount), then split the
      **available MDF fairly** (equal split) across them, respecting caps.
      Compute each selected partner's expected pipeline uplift and show its baseline.

    Steps:
      1) Align grants to outcomes using the learned lag.
      2) Aggregate per-partner total_amount and total_pipeline; ROI = pipeline / amount.
      3) Pick top-K partners by ROI (desc) among those with amount > 0.
      4) Equal-split the total_budget across those partners subject to caps.
      5) Uplift = allocation * ROI.
      6) Include partner baseline for the *impact month* (same as campaign layer).
         (We’ll pull the impact month outside and merge before reporting.)

    Returns:
      DataFrame with columns (for reporting downstream):
        partner, roi_partner, cap_monthly, allocation_partner, uplift_partner
      (Baseline column will be merged later for the impact month.)
    """
    # 1) Align grants to outcomes at month+lag
    grants_df = grants_df.copy()
    outcomes_df = outcomes_df.copy()
    grants_df["outcome_month"] = grants_df["month"] + pd.DateOffset(months=lag)

    merged = pd.merge(
        grants_df,
        outcomes_df,
        left_on=["outcome_month", "partner"],
        right_on=["month", "partner"],
        suffixes=("_grant", "_outcome"),
    )

    # Take the last known cap per partner (if multiple rows).
    caps = (
        merged.groupby("partner")["cap_monthly"]
        .agg(lambda x: x.dropna().iloc[-1] if len(x.dropna()) else np.nan)
        .reset_index()
    )

    # 2) Partner-level ROI (robust to division by zero).
    agg = merged.groupby("partner").agg(
        total_amount=("amount", "sum"),
        total_pipeline=("pipeline", "sum"),
    ).reset_index()
    agg = agg[agg["total_amount"] > 0].copy()
    if agg.empty:
        # Nothing to allocate if no partner has spend > 0; return empty table.
        return pd.DataFrame(columns=["partner", "roi_partner", "cap_monthly", "allocation_partner", "uplift_partner"])

    agg["roi_partner"] = agg["total_pipeline"] / agg["total_amount"]

    # 3) Top-K by ROI
    agg = agg.sort_values("roi_partner", ascending=False).reset_index(drop=True)
    top_k = max(1, min(top_k, len(agg)))
    top = agg.head(top_k).merge(caps, on="partner", how="left")
    top["cap_monthly"] = pd.to_numeric(top["cap_monthly"], errors="coerce").fillna(1e9)

    # 4) Equal-split allocation respecting caps. Simple, predictable.
    equal_share = total_budget / top_k
    top["allocation_partner"] = np.minimum(equal_share, top["cap_monthly"])

    # Redistribute leftover equally among those still under cap (small fixed passes).
    for _ in range(3):
        leftover = total_budget - top["allocation_partner"].sum()
        if leftover <= 1e-6:
            break
        room = (top["cap_monthly"] - top["allocation_partner"]).clip(lower=0)
        eligible = room > 0
        if not eligible.any():
            break
        add_each = leftover / eligible.sum()
        top.loc[eligible, "allocation_partner"] += np.minimum(add_each, room[eligible])

    # 5) Expected pipeline uplift per partner.
    top["uplift_partner"] = (top["allocation_partner"] * top["roi_partner"]).round(0)
    top["allocation_partner"] = top["allocation_partner"].round(0)

    return top[["partner", "roi_partner", "cap_monthly", "allocation_partner", "uplift_partner"]]


# --- LLM narrative (extended to include the partner fairness layer) ------------
def gen_llm_narrative(
    lag,
    total_budget,
    total_uplift_pipeline,
    impact_month,
    reco_df,                # campaign-level table
    partner_reco_df,        # NEW: partner-level fairness table
    top_k                   # NEW: K used for fairness
):
    """
    We now pass BOTH layers:
      - campaign-level (explore/exploit across campaign types)
      - partner-level fairness split across top-K partners

    The LLM is asked to add a short paragraph explaining the partner fairness choice.
    If LLM is unavailable, a deterministic text is returned (kept simple).
    """
    # Campaign top contributors (for context bullets)
    tmp = reco_df.copy()
    tmp["exp_won"] = tmp["allocation"] * (tmp["uplift"] / tmp["allocation"]).replace([np.inf, -np.inf], 0).fillna(0)
    top_camp = tmp.sort_values("uplift", ascending=False).head(3)[["campaign_type", "allocation", "uplift"]]

    # Partner JSON payload (compact)
    partner_payload = partner_reco_df[["partner", "allocation_partner", "uplift_partner"]].rename(
        columns={
            "allocation_partner": "mdf_allocation",
            "uplift_partner": "pipeline_uplift"
        }
    ).to_dict(orient="records")

    facts = {
        "lag_months": int(lag),
        "total_budget": round(float(total_budget), 2),
        "impact_month": impact_month.strftime("%Y-%m"),
        "total_uplift_pipeline": round(float(total_uplift_pipeline), 2),
        "campaign_top": [
            {
                "campaign_type": str(r.campaign_type),
                "mdf_allocation": round(float(r.allocation), 2),
                "pipeline_uplift": round(float(r.uplift), 2),
            }
            for r in top_camp.itertuples(index=False)
        ],
        "partner_fairness": {
            "k": int(top_k),
            "split": partner_payload  # [{partner, mdf_allocation, pipeline_uplift}, ...]
        },
    }

    # Prompt: politely tell the LLM to add a partner-fairness paragraph.
    prompt = f"""
Act as a marketing analyst summarizing an MDF recommendation for executives.

Context (JSON):
{json.dumps(facts, indent=2)}

Write 2–3 short paragraphs:
- Para 1: budget, learned lag (spend now → impact later), total pipeline uplift, and impact month.
- Para 2: briefly justify the campaign-type allocation (what worked; exploration kept simple).
- Para 3: explain the partner-fairness layer: why we split across top-{top_k} partners, how it balances ROI with fairness, and the expected uplift from those partners.

Keep it plain, concise, and free of jargon.
"""

    # Use LLM if available
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if genai and api_key:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(LLM_MODEL)
            response = model.generate_content(prompt)
            txt = (response.text or "").strip()
            if txt:
                return txt
        except Exception:
            pass

    # Fallback deterministic narrative
    lines = [
        f"We learned a {facts['lag_months']}-month delay from MDF to revenue; the uplift lands in {facts['impact_month']}.",
        f"Total budget is ${facts['total_budget']:,.0f}; expected pipeline uplift ≈ ${facts['total_uplift_pipeline']:,.0f}.",
        "Campaign allocation focuses on the strongest historical returns while preserving a small exploration slice for learning.",
        f"For partner fairness, we split funds across the top-{top_k} partners by historical ROI to balance performance with risk concentration.",
        "This keeps the plan understandable and equitable while still targeting high-return channels and partners.",
    ]
    return "\n".join(lines)


# --- PDF report (adds a second table for partners + nicer headers) -------------
def _format_money(x): return f"${x:,.0f}"
def _format_pct(x):   return f"{x:.1%}"
def _format_num(x):   return f"{x:.2f}"

def create_pdf_report(
    reco_df,                # campaign allocation
    partner_reco_df,        # NEW: partner fairness allocation
    baseline_df,            # for showing partner baseline at impact month
    impact_month,
    lag,
    total_uplift,
    narrative,
    plot_path
):
    """
    Builds a single PDF:
      - Title + narrative
      - Summary metrics
      - Table 1: Recommended campaign allocation (with NATURAL HEADERS)
      - Table 2: Partner fairness allocation (with NATURAL HEADERS + baseline)
      - Plot: Baseline vs With-MDF
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    doc = SimpleDocTemplate(OUTPUT_PDF, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Title (kept as in your file: "h1" style)
    story.append(Paragraph("MDF Allocation Recommendation", styles["h1"]))
    story.append(Spacer(1, 0.5 * cm))

    # Narrative
    story.append(Paragraph(narrative.replace("\n", "<br/>"), styles["BodyText"]))
    story.append(Spacer(1, 0.8 * cm))

    # Key Metrics
    story.append(Paragraph("Summary", styles["h2"]))
    key_metrics = [
        ["Estimated Impact Lag:", f"{lag} months"],
        ["Total Pipeline Uplift:", f"{_format_money(total_uplift)} (in {impact_month.strftime('%B %Y')})"],
    ]
    t_sum = Table(key_metrics, colWidths=[5 * cm, None])
    t_sum.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t_sum)
    story.append(Spacer(1, 0.8 * cm))

    # ----------------------------
    # Table 1 — Campaign allocation (natural headers)
    # ----------------------------
    story.append(Paragraph("Recommended MDF by Campaign Type", styles["h2"]))

    # Build a display copy to format & map headers:
    disp = reco_df.copy()
    disp = disp[["campaign_type", "prob_best", "allocation", "uplift"]]
    disp["prob_best"] = disp["prob_best"].apply(_format_pct)
    disp["allocation"] = disp["allocation"].apply(_format_money)
    disp["uplift"] = disp["uplift"].apply(_format_money)

    # Natural headers per your request:
    header_campaign = ["Campaign type", "Probability", "MDF allocation", "Pipeline uplift"]
    data_campaign = [header_campaign] + disp.values.tolist()

    t_campaign = Table(data_campaign, colWidths=[5*cm, 3*cm, 4*cm, 4*cm])
    t_campaign.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.royalblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (0, 0), "LEFT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(t_campaign)
    story.append(Spacer(1, 0.8 * cm))

    # ----------------------------
    # Table 2 — Partner fairness allocation (natural headers + baseline)
    # ----------------------------
    story.append(Paragraph("Partner Allocation (Fair Split Across Top Performers)", styles["h2"]))

    # Merge partner baseline for the impact month
    base_m = baseline_df.copy()
    base_m = base_m[base_m["month"] == impact_month][["partner", "baseline_pipeline"]]
    partner_disp = partner_reco_df.merge(base_m, on="partner", how="left")
    partner_disp["baseline_pipeline"] = partner_disp["baseline_pipeline"].fillna(0.0)

    # Keep & format only the requested fields with natural headers
    partner_disp = partner_disp[["partner", "baseline_pipeline", "allocation_partner", "uplift_partner"]].copy()
    partner_disp["baseline_pipeline"] = partner_disp["baseline_pipeline"].apply(_format_money)
    partner_disp["allocation_partner"] = partner_disp["allocation_partner"].apply(_format_money)
    partner_disp["uplift_partner"] = partner_disp["uplift_partner"].apply(_format_money)

    header_partner = ["Partner", "Baseline pipeline", "MDF allocation", "Pipeline uplift"]
    data_partner = [header_partner] + partner_disp.values.tolist()

    t_partner = Table(data_partner, colWidths=[5*cm, 4*cm, 4*cm, 4*cm])
    t_partner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkgreen),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (0, 0), "LEFT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(t_partner)
    story.append(Spacer(1, 0.8 * cm))

    # Plot
    story.append(Paragraph("Forecast Impact", styles["h2"]))
    if os.path.exists(plot_path):
        img = Image(plot_path, width=18 * cm, height=11 * cm)
        img.hAlign = "CENTER"
        story.append(img)
    else:
        story.append(Paragraph("Plot image not found.", styles["BodyText"]))

    doc.build(story)
    print(f"PDF report saved to: {OUTPUT_PDF}")


# --- Main Orchestration (kept, with partner layer inserted) --------------------
def main():
    ensure_files_exist()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load
    F = load_grants_csv(FILE_GRANTS)     # grants: month, partner, campaign_type, amount
    G = load_outcomes_csv(FILE_OUTCOMES) # outcomes: month, partner, pipeline, closed_won, tier, cap_monthly
    B = load_baseline_csv(FILE_BASELINE) # baseline: month, partner, baseline_pipeline

    # Coerce dates
    F["month"] = coerce_month(F["month"])
    G["month"] = coerce_month(G["month"])
    B["month"] = coerce_month(B["month"])

    # Current planning month = max month of grants
    current_month = F["month"].max()

    # 1) Learn lag
    lag = estimate_lag(F, G, lags=(2, 3))

    # 2) Campaign-type posteriors + allocation (your logic)
    post = roi_posteriors(F, G, lag)
    reco = thompson_allocate(post, total_budget=TOTAL_BUDGET, explore_frac=EXPLORE_FRAC, seed=42)

    # 3) Impact on baseline (campaign layer)
    with_mdf_series, total_uplift_pipeline, impact_month = apply_uplift_to_baseline(
        baseline_next3=B,
        current_month=current_month,
        lag=lag,
        reco=reco,
        close_rate=CLOSE_RATE
    )

    # 4) Plot
    make_impact_plot(B, with_mdf_series, out_path=OUTPUT_PLOT)

    # 5) NEW: Partner fairness layer (top-K equal split under caps)
    partner_reco = partner_topk_fair_allocation(
        grants_df=F,
        outcomes_df=G,
        baseline_df=B,
        lag=lag,
        total_budget=TOTAL_BUDGET,
        top_k=TOP_PARTNER_COUNT,
    )

    # 6) Narrative (now includes partner fairness explanation)
    narrative = gen_llm_narrative(
        lag=lag,
        total_budget=TOTAL_BUDGET,
        total_uplift_pipeline=total_uplift_pipeline,
        impact_month=impact_month,
        reco_df=reco,
        partner_reco_df=partner_reco,
        top_k=TOP_PARTNER_COUNT
    )

    # 7) PDF report with TWO tables and natural headers
    create_pdf_report(
        reco_df=reco,
        partner_reco_df=partner_reco,
        baseline_df=B,
        impact_month=impact_month,
        lag=lag,
        total_uplift=total_uplift_pipeline,
        narrative=narrative,
        plot_path=OUTPUT_PLOT,
    )

    # Save campaign-type recommendations to CSV (kept, unchanged)
    reco.to_csv(OUTPUT_RECO, index=False)
    print(f"Recommendations saved to: {OUTPUT_RECO}")
    print(
        f"Done.\nLag: {lag} months | Impact month: {impact_month.strftime('%Y-%m')} "
        f"| Expected pipeline uplift: ${total_uplift_pipeline:,.0f}"
    )


if __name__ == "__main__":
    main()
