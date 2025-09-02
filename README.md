# Agentic MDF Optimizer — Lag-Aware, Fair, LLM-Explained

> This is an **agentic AI architecture** for joint ownership: a small, agentic loop that learns the delay, allocates under constraints, and explains itself—so Sales and Marketing can sign one plan and be measured against one pipeline.

* * *

## ✨ Why This Matters
- MDF is tracked, not optimized. This proposes **where to place the next dollar**, not just what happened.
- **Shared ownership:** one pipeline number both Sales and Marketing can sign.
- **Explainability:** the LLM produces an exec-ready rationale; every number is audit-friendly.

---

## Business Goals
- Lift **pipeline/revenue per $MDF** vs. last 3 months.
- Allocate under **policy caps**; avoid over-concentration via **top-K partner fairness**.
- Show **baseline → with-MDF** impact at the month where uplift lands (lag-aware).

## Demo assets

- 📄 **Test run (PDF):** [Agentic AI MDF Allocation recomender](https://github.com/AIArpi/mdf_allocation_recommender/blob/main/output/report.pdf) — a captured report for quick review.

* * *

## ⚙️ Architecture (at a glance)

<!-- Architecture (at a glance) -->
<pre>
Grants (spend) + Outcomes (pipeline/caps) + Baseline (next 3 mo)
│
├── Lag Learner (finds delay: spend t → impact t+L)
├── ROI Learner (Gamma posterior per campaign type)
├── Campaign Allocator (Thompson: explore + exploit)
├── Partner Fairness (equal split across top-K ROI partners, caps)
├── Guardrails (caps, sum=B, anomaly flags)
└── Reviewer (LLM, gemma-3-27b-it) → PDF narrative + "scale/stop"
└── Outputs: PDF + CSV + impact plot
</pre>

---

## Features
- **Lag-aware uplift:** impact added at *m+L* (usually 2–3 months).
- **Two layers of allocation:**  
  1) **Campaign type** — probability of being best, MDF allocation, expected uplift.  
  2) **Partner fairness** — split the pot across **top-K** partners by historical ROI (K=2 default), under caps.
- **Executive report (PDF):** narrative + 2 tables + baseline vs. with-MDF chart.
- **Downloadables:** `output/recommendations.csv`, `output/report.pdf`, `output/plot.png`.

---

## Screens (what you’ll see)
- **Table 1 (Campaign type):** *Campaign type · Probability · MDF allocation · Pipeline uplift*  
- **Table 2 (Partner fairness):** *Partner · Baseline pipeline · MDF allocation · Pipeline uplift*  
- **Chart:** *Baseline vs. With-MDF* (impact month highlighted in the PDF narrative)

---

## KPIs (tracked)
- Pipeline (or revenue) **per $MDF** vs. last 3 months.  
- **% within policy** (caps, floors).  
- **Concentration** (share to top partner), bounded by K-split.  
- **Uncertainty trend** (posterior variance) — should decline as data accrues.

* * *

## Getting Started

### 0) Prerequisites
- Python **3.10+**
- Google API key (for the narrative): default model **`gemma-3-27b-it`**

### 1) Setup
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Data

Place the three CSVs under ./data/ with exact names:

```
data/
  mdf_outcomes_sample.csv    # GRANTS: month, partner, campaign_type, amount 
  mdf_grants_sample.csv      # OUTCOMES: month, partner, pipeline, closed_won, tier, cap_monthly
  baseline_pipeline_next3.csv# BASELINE: month, partner, baseline_pipeline
```

### 3) Configure LLM (optional but recommended)

**# Windows PowerShell**
```bash
setx GOOGLE_API_KEY "your-api-key-here"
```
**# macOS / Linux**
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

### 4) Run
```bash
python app.py
```

### Outputs:
```
output/
  plot.png
  recommendations.csv
  report.pdf
```

---

## Configuration

**Total budget and exploration % are set in `app.py`:**
- `TOTAL_BUDGET = 15000`
- `EXPLORE_FRAC = 0.25`

**Top-K partners for fairness:**
- `TOP_PARTNER_COUNT = 2` (set to 3 for a broader split)

**Close rate for context in report:**
- `CLOSE_RATE = 0.20`

**Default LLM model:**
- `LLM_MODEL = "gemma-3-27b-it"`

---

## Project Structure
```
agentic-mdf-optimizer/
  app.py                  # main script, generates CSV/plot/PDF
  requirements.txt
  data/
    mdf_outcomes_sample.csv
    mdf_grants_sample.csv
    baseline_pipeline_next3.csv
  output/
    plot.png
    recommendations.csv
    report.pdf
```

## License

MIT — see [`LICENSE`](LICENSE).

## Author

**Arpad Bihami** — [LinkedIn](https://www.linkedin.com/in/arpadbihami)

> Hiring‑manager note: this project demonstrates an **agentic AI architecture** that ties MDF to a **single, shared pipeline** — with guardrails, fairness, and an executive narrative that explains the *why*, not just the numbers.
