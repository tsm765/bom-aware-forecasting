"""
BOM-Aware Demand Forecasting Under Component Churn
===================================================
Single-file Streamlit dashboard.
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
from pathlib import Path
from PIL import Image

# ── paths & constants ──────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent
DATA_DIR  = ROOT / "data"
PLOTS_DIR = ROOT / "plots"

BASE_UNIT_COST   = 200
SAVING_PER_EVENT = 4187 / 26        # = 161.04
EVENTS_PER_YEAR  = 26
Z                = 1.65
MAE_NAIVE        = 12.84
MAE_BOM          = 11.35
DELTA_MAE        = MAE_NAIVE - MAE_BOM
LT_MONTHS        = 6 / (52 / 12)
HOLDING_RATE     = 0.22
ORDERING_COST    = 100
P_STOCKOUT       = 0.05

def compute_saving(unit_cost):
    c1 = Z * DELTA_MAE * (LT_MONTHS ** 0.5) * unit_cost * HOLDING_RATE * EVENTS_PER_YEAR
    c2 = (MAE_NAIVE / MAE_BOM - 1) * ORDERING_COST * EVENTS_PER_YEAR
    c3 = DELTA_MAE * P_STOCKOUT * LT_MONTHS * unit_cost * EVENTS_PER_YEAR
    return dict(c1=c1, c2=c2, c3=c3, total=c1 + c2 + c3)

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG — must be first Streamlit call
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="BOM-Aware Demand Forecasting",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS — injected before any page content.
#  Uses var(--text-color) / var(--secondary-background-color) from
#  Streamlit's own theme so both light and dark modes work without
#  hardcoded colour overrides.
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] li {
    font-size: 1.03rem;
    line-height: 1.8;
    color: var(--text-color);
}
.page-title {
    padding-bottom: 0.45rem;
    border-bottom: 3px solid #0D9488;
    margin-bottom: 1.5rem;
    color: var(--text-color);
}
.sdiv {
    border: none;
    border-top: 1px solid rgba(128,128,128,0.2);
    margin: 2rem 0;
}
.m-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(13,148,136,0.3);
    border-radius: 10px;
    padding: 1rem 0.75rem;
    text-align: center;
}
.m-card .m-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-color);
    opacity: 0.6;
    margin-bottom: 6px;
}
.m-card .m-val { font-size: 1.9rem; font-weight: 700; line-height: 1.1; }
.m-card .m-val.teal { color: #0D9488; }
.m-card .m-val.blue { color: #2563EB; }
.callout {
    background: rgba(37,99,235,0.1);
    border-left: 4px solid #2563EB;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.25rem;
    margin: 1.25rem 0;
}
.callout p { margin: 0; font-size: 0.97rem; color: var(--text-color); }
.closing-q {
    text-align: center;
    font-size: 1.3rem;
    font-weight: 500;
    color: var(--text-color);
    line-height: 1.7;
    padding: 2.5rem 1.5rem;
    border-top: 2px solid rgba(128,128,128,0.2);
    margin-top: 2.5rem;
}
.g-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.18);
    border-radius: 10px;
    padding: 1.1rem 1.25rem;
}
.saving-hl {
    text-align: center;
    padding: 1.75rem 1rem;
    background: rgba(13,148,136,0.1);
    border: 1px solid rgba(13,148,136,0.35);
    border-radius: 14px;
    margin: 1.25rem 0;
}
.saving-hl .s-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    opacity: 0.6;
    color: var(--text-color);
    margin-bottom: 10px;
}
.saving-hl .s-amount { font-size: 3.5rem; font-weight: 700; color: #0D9488; line-height: 1; }
.saving-hl .s-desc { font-size: 1rem; color: var(--text-color); opacity: 0.8; margin-top: 10px; }
.saving-hl .s-rate { font-size: 0.85rem; color: var(--text-color); opacity: 0.5; margin-top: 5px; }
.p-table, .c-table { width: 100%; border-collapse: collapse; font-size: 0.91rem; }
.p-table th, .c-table th {
    padding: 10px 13px;
    font-weight: 600;
    border-bottom: 2px solid rgba(128,128,128,0.25);
    color: var(--text-color);
    background: var(--secondary-background-color);
    text-align: left;
}
.c-table th { text-align: right; }
.c-table th:first-child { text-align: left; }
.p-table td, .c-table td {
    padding: 10px 13px;
    border-bottom: 1px solid rgba(128,128,128,0.12);
    color: var(--text-color);
    vertical-align: top;
}
.p-table td:nth-child(2) { color: #0D9488; font-weight: 600; }
.c-table td { text-align: right; }
.c-table td:first-child { text-align: left; font-weight: 500; }
.c-table tr.base-row td { background: rgba(13,148,136,0.1); color: #0D9488; font-weight: 600; }
.muted {
    font-size: 0.82rem;
    color: var(--text-color);
    opacity: 0.5;
    font-style: italic;
    line-height: 1.6;
    margin-top: 0.6rem;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
<div style="padding:0.4rem 0 1.25rem">
  <div style="font-size:1.05rem;font-weight:700;color:var(--text-color);line-height:1.4">
    BOM-Aware<br>Demand Forecasting
  </div>
  <div style="font-size:0.78rem;opacity:0.5;color:var(--text-color);margin-top:4px">
    Under Component Churn
  </div>
</div>
""", unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        options=[
            "1 · Problem statement",
            "2 · Data overview",
            "3 · Model comparison",
            "4 · Cost simulation",
            "5 · Conclusion",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
<div style="font-size:0.75rem;opacity:0.45;color:var(--text-color);line-height:1.7">
    Synthetic dataset · 4 product lines<br>
    48 component families · 77 substitutions<br>
    36 months · SES baseline model<br><br>
    <em>All data is synthetic. Normet references
    use publicly available information only.</em>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def sdiv():
    st.markdown('<hr class="sdiv">', unsafe_allow_html=True)

def callout_box(text):
    st.markdown(f'<div class="callout"><p>{text}</p></div>', unsafe_allow_html=True)

def page_header(title):
    st.markdown(f'<h1 class="page-title">{title}</h1>', unsafe_allow_html=True)

def img(fname):
    st.image(Image.open(PLOTS_DIR / fname), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — PROBLEM STATEMENT
# ══════════════════════════════════════════════════════════════════════════════
def page_1():
    page_header("The problem with part numbers")

    st.markdown("""
Most forecasting systems are built around part numbers. When a component gets substituted,
the new part number starts fresh. No history, no trend, no seasonal pattern. For the next
few months, the system is essentially making its best guess based on very little information.
That gap might seem small in isolation. Across a portfolio of components going through
regular lifecycle transitions, it adds up.
""")

    st.markdown("""
For manufacturers running monthly S&OP cycles, this matters for a specific reason.
The material demand forecast that drives procurement does not come from a crystal ball.
It comes from a supply plan, which comes from a demand review, which is broken down
through the BOM via MRP. That chain is only as good as the demand signals at the
bottom of it. When a component substitution breaks the demand signal, the error does
not stay local. It moves up the chain.
""")

    st.markdown("""
This is a fairly common situation for any manufacturer managing product generations,
ramp-ups, and ramp-downs simultaneously. Engineering changes happen. Suppliers get
qualified and disqualified. New models replace old ones. Each of those events is a
substitution event, and each one creates a short window where the forecasting system
is working with less information than it needs.
""")

    sdiv()

    st.markdown("#### What a substitution event looks like to a forecasting system")

    st.markdown("""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:1.25rem;margin:1rem 0 1.5rem">

  <div style="border:1px solid rgba(220,38,38,0.35);border-radius:10px;overflow:hidden">
    <div style="background:rgba(220,38,38,0.1);padding:0.75rem 1.1rem;
                border-bottom:1px solid rgba(220,38,38,0.2)">
      <span style="font-size:0.88rem;font-weight:600;color:#f87171">
        Standard forecasting — part number view
      </span>
    </div>
    <div style="padding:1rem 1.1rem;background:var(--secondary-background-color)">
      <div style="font-family:monospace;font-size:0.82rem;
                  color:var(--text-color);line-height:2.2">
        <span style="opacity:0.5">Jan–Aug 2022 </span>PN-A001
        <span style="color:#22c55e"> ▓▓▓▓▓▓▓▓</span> (8 months)<br>
        <span style="color:#f87171;font-weight:600">Sep 2022 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span style="color:#f87171"> ↓ substitution</span><br>
        <span style="opacity:0.5">Sep 2022+ &nbsp;&nbsp;&nbsp;</span>PN-B002
        <span style="color:#93c5fd"> ▓</span>
        <span style="opacity:0.4"> &nbsp;??? forecast</span>
      </div>
      <p style="font-size:0.84rem;color:#f87171;margin:0.75rem 0 0;
                background:rgba(220,38,38,0.08);border-radius:6px;
                padding:0.6rem 0.8rem;border:1px solid rgba(220,38,38,0.15)">
        The new part number has no history. The model has nothing to work with
        during the most critical window.
      </p>
    </div>
  </div>

  <div style="border:1px solid rgba(13,148,136,0.35);border-radius:10px;overflow:hidden">
    <div style="background:rgba(13,148,136,0.1);padding:0.75rem 1.1rem;
                border-bottom:1px solid rgba(13,148,136,0.2)">
      <span style="font-size:0.88rem;font-weight:600;color:#2dd4bf">
        BOM-aware forecasting — functional ID view
      </span>
    </div>
    <div style="padding:1rem 1.1rem;background:var(--secondary-background-color)">
      <div style="font-family:monospace;font-size:0.82rem;
                  color:var(--text-color);line-height:2.2">
        <span style="opacity:0.5">Jan–Aug 2022 </span>PN-A001
        <span style="color:#22c55e"> ▓▓▓▓▓▓▓▓</span><br>
        <span style="color:#2dd4bf;font-weight:600">Sep 2022 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span style="color:#2dd4bf"> ↓ chained under FID-01</span><br>
        <span style="opacity:0.5">Sep 2022+ &nbsp;&nbsp;&nbsp;</span>PN-B002
        <span style="color:#22c55e"> ▓▓▓▓▓▓▓▓</span><span style="color:#2dd4bf">▓▓▓</span>
      </div>
      <p style="font-size:0.84rem;color:#2dd4bf;margin:0.75rem 0 0;
                background:rgba(13,148,136,0.08);border-radius:6px;
                padding:0.6rem 0.8rem;border:1px solid rgba(13,148,136,0.15)">
        Full history preserved through the functional ID. The model sees 33 months
        of demand, not 1.
      </p>
    </div>
  </div>

</div>
""", unsafe_allow_html=True)

    sdiv()

    st.markdown("""
<div class="closing-q">
    It is worth asking how well your current forecasting setup handles
    the moments when the BOM changes.
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — DATA OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
def page_2():
    page_header("The dataset")

    st.markdown("""
This analysis uses a synthetic dataset built to reflect the dynamics of real manufacturing
environments: four product lines, 48 component families, 125 part numbers, 36 months of
monthly demand, and 77 substitution events spread across the dataset.
""")

    st.markdown("""
The dataset was designed rather than sourced. Real BOM demand data is proprietary. But
designing the data means encoding what is actually known about how these systems behave:
that demand for a function continues even when the part number serving that function
changes, that January often brings a procurement surge driven by budget cycles, that
substitution events cluster around generation transitions. Building a synthetic dataset
that reflects these patterns and then testing whether the model can recover them is a
more transparent form of validation than fitting to one company's historical data.
""")

    st.markdown("""
<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:1.5rem 0">
  <div class="m-card"><div class="m-label">Product lines</div>
    <div class="m-val teal">4</div></div>
  <div class="m-card"><div class="m-label">Component families</div>
    <div class="m-val teal">48</div></div>
  <div class="m-card"><div class="m-label">Part numbers</div>
    <div class="m-val teal">125</div></div>
  <div class="m-card"><div class="m-label">Months</div>
    <div class="m-val teal">36</div></div>
  <div class="m-card"><div class="m-label">Substitution events</div>
    <div class="m-val teal">77</div></div>
</div>
""", unsafe_allow_html=True)

    sdiv()

    st.markdown("#### What a substitution event actually looks like in the data")
    st.markdown("""
The chart below shows the same component family from two perspectives simultaneously.
The top panel is what a standard forecasting system sees: three independent time series,
each starting from scratch when a new part number takes over. The bottom panel is what
the BOM-aware model sees: one continuous signal, uninterrupted across the substitution
events. The underlying demand is identical. Only the framing changes.
""")
    img("fig1_substitution_event_naive_vs_bomaware.png")

    sdiv()

    st.markdown("#### Seasonality in the dataset")
    st.markdown("""
January consistently shows elevated demand across all four product lines. This reflects
a pattern common in industrial procurement: budget cycles reset at the start of the year
and maintenance planning for the coming twelve months typically drives an early ordering
surge. The dataset encodes a 20 to 40 percent January premium, visible in the heatmap
below. Any forecasting model applied to this data needs to handle this pattern correctly
to produce useful predictions.
""")
    img("fig2_january_seasonality_heatmap.png")

    sdiv()

    st.markdown("#### Choosing a baseline model")
    st.markdown("""
Before settling on a baseline model to compare against, two candidate approaches were
evaluated. ARIMA, the standard algorithmic choice in industrial forecasting, turned out
to have a fundamental problem for this specific use case: it needs a minimum amount of
history to fit reliably. In the transition window, where history is exactly what is
missing, ARIMA simply cannot produce a forecast for the majority of components.
""")
    st.markdown("""
SES, which has no minimum history requirement and reflects how most ERP systems handle
short-history components in practice, is the more honest and more challenging baseline
to beat. The chart below shows the comparison. The 16 percent fit rate for ARIMA is not
a result of algorithmic weakness — it is a structural consequence of the problem itself.
Any method that requires sufficient prior history will fail in the same way.
""")
    img("fig9_arima_vs_ses_comparison.png")

    sdiv()

    st.markdown("#### Distribution of substitution events across the dataset")
    c1, c2 = st.columns([2, 1])
    with c1:
        img("fig4_substitution_event_timeline.png")
    with c2:
        img("fig8_substitution_reasons.png")

    st.markdown("""
The 77 substitution events are spread across all four product lines and distributed
throughout the 36-month period, with a minimum spacing of six months between events
within any single component family. The reasons for substitution in the dataset reflect
the real mix of drivers: end-of-life discontinuations, cost reduction decisions, design
changes, and supply disruptions.
""")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
def page_3():
    page_header("Naive vs BOM-aware: the comparison")

    st.markdown("""
The comparison is straightforward. Same algorithm, same parameter selection process,
same train-test split. The only difference is what each model sees as its input. The
naive model sees the part number history in isolation. The BOM-aware model sees the
full functional demand history, chained across all part numbers that have served that
role. The gap between them is largest exactly where it matters most, in the first few
months after a substitution event.
""")

    st.markdown("""
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:1.5rem 0">
  <div class="m-card">
    <div class="m-label">Naive MAPE — overall</div>
    <div class="m-val blue">14.5%</div>
  </div>
  <div class="m-card">
    <div class="m-label">BOM-aware MAPE — overall</div>
    <div class="m-val teal">11.2%</div>
  </div>
  <div class="m-card">
    <div class="m-label">Naive MAPE — transition</div>
    <div class="m-val blue">11.5%</div>
  </div>
  <div class="m-card">
    <div class="m-label">BOM-aware MAPE — transition</div>
    <div class="m-val teal">8.8%</div>
  </div>
</div>
""", unsafe_allow_html=True)

    sdiv()

    st.markdown("#### Four-panel comparison")
    img("fig11_naive_vs_bom_full.png")

    st.markdown("""
The top-left panel shows overall MAPE by product line. The improvement is consistent
across all four lines, which matters: a finding that only appears in one product line
could be a data artefact. The 3.3 percentage point improvement showing up consistently
across Alpha, Beta, Gamma and Delta suggests it is structural, not incidental.
""")
    st.markdown("""
The top-right scatter plot shows every component family individually. Points below the
diagonal are families where BOM-aware forecasting wins. The majority sit there. The
few exceptions are families where the naive model happens to have a longer series in
the test window, which narrows the gap between the two approaches.
""")

    sdiv()

    st.markdown("#### The transition window in detail")
    st.markdown("""
The bottom two panels are the centrepiece of this analysis. They show MAPE and MAE
month by month across the 0 to 6 month window following a substitution event. This
is where the two models diverge most clearly, and where the divergence is most
consequential for real procurement decisions.
""")

    callout_box(
        "One internal check worth noting. The BOM-aware models averaged a smoothing "
        "parameter of 0.30, compared to 0.40 for the naive models. A lower value means "
        "the model is leaning more on historical patterns rather than recent observations. "
        "That is exactly what you would expect when the training series is 33 months "
        "long instead of 8. The mathematics is behaving the way the theory says it should."
    )

    st.markdown("""
At five and six months post-substitution, the MAE results are mixed. This is expected.
As time passes after a substitution event, the naive model accumulates its own history
and the gap between the two approaches naturally narrows. The window where BOM-aware
forecasting makes the biggest difference is the acute early period, the first two to
four months after a substitution, and that is where the improvement is most consistent
and most consequential for procurement decisions.
""")

    sdiv()

    st.markdown("#### Why the 11.7% improvement rate matters")
    st.markdown("""
The improvement in forecast accuracy is not sensitive to unit cost, product line, or
volume. It is a ratio: the BOM-aware model reduces forecast error by 11.7 percent
relative to the naive baseline, and that ratio holds across all conditions tested.
This invariance is what makes the cost projections on the next page defensible.
""")

    st.markdown("""
<div class="g-card" style="margin:1rem 0">
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;text-align:center">
    <div>
      <div style="font-size:0.78rem;opacity:0.55;color:var(--text-color);margin-bottom:4px">PL-Alpha</div>
      <div style="font-size:1.5rem;font-weight:700;color:#0D9488">−4.2pp</div>
    </div>
    <div>
      <div style="font-size:0.78rem;opacity:0.55;color:var(--text-color);margin-bottom:4px">PL-Beta</div>
      <div style="font-size:1.5rem;font-weight:700;color:#0D9488">−3.1pp</div>
    </div>
    <div>
      <div style="font-size:0.78rem;opacity:0.55;color:var(--text-color);margin-bottom:4px">PL-Gamma</div>
      <div style="font-size:1.5rem;font-weight:700;color:#0D9488">−2.8pp</div>
    </div>
    <div>
      <div style="font-size:0.78rem;opacity:0.55;color:var(--text-color);margin-bottom:4px">PL-Delta</div>
      <div style="font-size:1.5rem;font-weight:700;color:#0D9488">−3.9pp</div>
    </div>
  </div>
  <div style="text-align:center;margin-top:0.75rem;font-size:0.79rem;
              opacity:0.45;color:var(--text-color)">
    MAPE reduction (naive → BOM-aware) by product line
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — COST SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
def page_4():
    page_header("From forecast error to inventory cost")

    st.markdown("""
Forecast error is not abstract. It translates directly into inventory decisions: how
much safety stock to hold, how often to reorder, and how often to absorb the cost of
a stockout or an emergency purchase. The calculations below convert the MAE improvement
from the model comparison into annual euro costs, using conservative assumptions
throughout.
""")

    sdiv()
    st.markdown("#### Assumptions")
    st.markdown("""
Every number in this analysis rests on a specific assumption. The table below lists
each one with a brief rationale. None of these figures are optimistic.
""")

    st.markdown("""
<table class="p-table">
  <thead>
    <tr>
      <th style="width:30%">Parameter</th>
      <th style="width:22%">Value</th>
      <th>Rationale</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Naive transition MAE</td><td>12.84 units/month</td>
        <td>Direct output from the SES baseline model on the held-out test set.</td></tr>
    <tr><td>BOM-aware transition MAE</td><td>11.35 units/month</td>
        <td>Direct output from the BOM-aware SES model on the same test set.</td></tr>
    <tr><td>Substitution events per year</td><td>26</td>
        <td>77 events over 36 months, annualised to 25.7 and rounded to 26.</td></tr>
    <tr><td>Service level</td><td>95% · Z = 1.65</td>
        <td>Standard for industrial spare parts and certified components.</td></tr>
    <tr><td>Lead time</td><td>6 weeks · 1.38 months</td>
        <td>Conservative for certified heavy equipment components with approved suppliers.</td></tr>
    <tr><td>Unit cost — baseline</td><td>€200</td>
        <td>Conservative for certified industrial equipment components. Sensitivity from €50 to €2,000 is provided.</td></tr>
    <tr><td>Holding cost rate</td><td>22% per year</td>
        <td>Standard operations management literature. Covers capital, storage, and obsolescence risk.</td></tr>
    <tr><td>Ordering cost per PO</td><td>€100</td>
        <td>Conservative. Emergency procurement of certified components typically involves expediting fees and supplier premiums well above this figure.</td></tr>
    <tr><td>Forecast error proxy for σ</td><td>MAE directly</td>
        <td>MAE is a reasonable approximation of forecast error standard deviation for symmetric distributions at this scale.</td></tr>
  </tbody>
</table>
""", unsafe_allow_html=True)

    sdiv()
    st.markdown("#### Three cost components")

    st.markdown("""
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin:1rem 0 1.5rem">

  <div class="g-card">
    <div style="font-size:0.73rem;text-transform:uppercase;letter-spacing:0.07em;
                opacity:0.55;color:var(--text-color);margin-bottom:8px">
      1 — SS holding cost
    </div>
    <div style="font-size:0.87rem;color:var(--text-color);line-height:1.65;opacity:0.85">
      The naive model's larger forecast error requires more safety stock to maintain
      the same 95% service level. Every extra unit held costs 22% of its value per year.
    </div>
    <div style="font-family:monospace;font-size:0.79rem;color:#0D9488;
                background:rgba(13,148,136,0.08);border-radius:6px;
                padding:0.5rem 0.75rem;margin-top:0.75rem">
      Z × ΔMAE × √LT × cost × rate × events
    </div>
  </div>

  <div class="g-card">
    <div style="font-size:0.73rem;text-transform:uppercase;letter-spacing:0.07em;
                opacity:0.55;color:var(--text-color);margin-bottom:8px">
      2 — Emergency ordering
    </div>
    <div style="font-size:0.87rem;color:var(--text-color);line-height:1.65;opacity:0.85">
      Higher forecast error drives proportionally more emergency reorders during
      the transition window. Extra POs scale with the relative MAE improvement ratio.
    </div>
    <div style="font-family:monospace;font-size:0.79rem;color:#0D9488;
                background:rgba(13,148,136,0.08);border-radius:6px;
                padding:0.5rem 0.75rem;margin-top:0.75rem">
      (MAE_naive / MAE_bom − 1) × PO_cost × events
    </div>
  </div>

  <div class="g-card">
    <div style="font-size:0.73rem;text-transform:uppercase;letter-spacing:0.07em;
                opacity:0.55;color:var(--text-color);margin-bottom:8px">
      3 — Stockout exposure
    </div>
    <div style="font-size:0.87rem;color:var(--text-color);line-height:1.65;opacity:0.85">
      With a 5% probability of a stockout cycle, the naive model's larger error pool
      produces more expected shortage units, each costed at emergency procurement price.
    </div>
    <div style="font-family:monospace;font-size:0.79rem;color:#0D9488;
                background:rgba(13,148,136,0.08);border-radius:6px;
                padding:0.5rem 0.75rem;margin-top:0.75rem">
      ΔMAE × P(stockout) × LT × cost × events
    </div>
  </div>

</div>
""", unsafe_allow_html=True)

    sdiv()
    st.markdown("#### Annual cost comparison")

    rows = [(uc, compute_saving(uc)) for uc in [100, 200, 500, 2000]]
    tbl  = """<table class="c-table">
  <thead>
    <tr>
      <th style="text-align:left">Unit cost</th>
      <th>SS holding saving</th>
      <th>Emergency PO saving</th>
      <th>Stockout saving</th>
      <th>Total annual saving</th>
    </tr>
  </thead>
  <tbody>"""
    for uc, r in rows:
        cls = ' class="base-row"' if uc == 200 else ""
        tbl += (f'\n    <tr{cls}>'
                f'<td>€{uc:,}</td>'
                f'<td>€{r["c1"]:,.0f}</td>'
                f'<td>€{r["c2"]:,.0f}</td>'
                f'<td>€{r["c3"]:,.0f}</td>'
                f'<td><strong>€{r["total"]:,.0f}</strong></td>'
                f'</tr>')
    tbl += "\n  </tbody>\n</table>"
    st.markdown(tbl, unsafe_allow_html=True)

    st.markdown("""
The emergency ordering saving of €341 per year is fixed regardless of unit cost — it
reflects the reduction in purchase order frequency alone. At €200 per unit, the safety
stock holding component drives 79 percent of the total saving.
""")

    sdiv()
    st.markdown("#### Sensitivity: annual saving across the €50–€2,000 unit cost range")
    img("fig12_cost_sensitivity.png")

    sdiv()
    st.markdown("#### What does this look like at your scale?")
    st.markdown("""
These are illustrative projections based on the improvement rate demonstrated in this
analysis. Actual figures depend on your portfolio composition and real procurement costs.
""")

    # ── live sliders ───────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        unit_cost_sel = st.slider(
            "Average unit cost (€)",
            min_value=50, max_value=2000, value=200, step=10, format="€%d"
        )
    with col_b:
        components_sel = st.slider(
            "Substitution-affected components per year",
            min_value=100, max_value=5000, value=1000, step=50
        )

    spe           = SAVING_PER_EVENT * (unit_cost_sel / BASE_UNIT_COST)
    annual_saving = spe * components_sel
    est_tag       = " — central estimate" if (unit_cost_sel == 200 and components_sel == 1000) else ""

    st.markdown(f"""
<div class="saving-hl">
  <div class="s-label">estimated annual saving — BOM-aware forecasting</div>
  <div class="s-amount">€{annual_saving:,.0f}</div>
  <div class="s-desc">{components_sel:,} components per year × €{unit_cost_sel:,} per unit{est_tag}</div>
  <div class="s-rate">€{spe:,.0f} saving per substitution event, derived from this analysis</div>
</div>
""", unsafe_allow_html=True)

    st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:0.5rem 0 1.5rem">
  <div class="g-card" style="text-align:center">
    <div style="font-size:0.75rem;opacity:0.55;color:var(--text-color);margin-bottom:4px">
      Conservative (500 components)
    </div>
    <div style="font-size:1.3rem;font-weight:600;color:var(--text-color)">€{500*spe:,.0f}</div>
  </div>
  <div style="background:rgba(13,148,136,0.1);border:1px solid rgba(13,148,136,0.3);
              border-radius:10px;padding:1.1rem 0.75rem;text-align:center">
    <div style="font-size:0.75rem;opacity:0.55;color:var(--text-color);margin-bottom:4px">
      Central (1,000 components)
    </div>
    <div style="font-size:1.3rem;font-weight:600;color:#0D9488">€{1000*spe:,.0f}</div>
  </div>
  <div class="g-card" style="text-align:center">
    <div style="font-size:0.75rem;opacity:0.55;color:var(--text-color);margin-bottom:4px">
      Optimistic (1,500 components)
    </div>
    <div style="font-size:1.3rem;font-weight:600;color:var(--text-color)">€{1500*spe:,.0f}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("**Reference table** — fixed combinations of unit cost and component count")
    ref = """<table class="c-table" style="font-size:0.88rem">
  <thead>
    <tr>
      <th style="text-align:left">Unit cost</th>
      <th>500 components</th>
      <th>1,000 components</th>
      <th>1,500 components</th>
    </tr>
  </thead>
  <tbody>"""
    for ruc in [100, 200, 500, 2000]:
        s    = SAVING_PER_EVENT * (ruc / BASE_UNIT_COST)
        cls  = ' class="base-row"' if ruc == 200 else ""
        ref += (f'\n    <tr{cls}>'
                f'<td>€{ruc:,}</td>'
                f'<td>€{500*s:,.0f}</td>'
                f'<td>€{1000*s:,.0f}</td>'
                f'<td>€{1500*s:,.0f}</td></tr>')
    ref += "\n  </tbody>\n</table>"
    st.markdown(ref, unsafe_allow_html=True)

    st.markdown("""
<p class="muted">
Projection based on 11.7% invariant forecast improvement rate and €161 per event saving
derived from the synthetic dataset analysis at €200 per unit. Actual saving depends on
real portfolio composition and procurement costs.
</p>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — CONCLUSION
# ══════════════════════════════════════════════════════════════════════════════
def page_5():
    page_header("Conclusion")

    st.markdown("### What this shows")
    st.markdown("""
A single change upstream, chaining demand histories under a functional ID before the
model runs, produces consistent improvement in transition-window forecast accuracy
across all product lines. The algorithm does not change. The MRP system does not change.
The BOM structure does not change. One additional data layer connects part number history
to functional demand, and the forecasting system stops going blind every time an
engineering change happens.
""")

    st.markdown("""
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin:1.5rem 0">
  <div class="g-card" style="text-align:center">
    <div style="font-size:2rem;font-weight:700;color:#0D9488;line-height:1">−3.3pp</div>
    <div style="font-size:0.82rem;color:var(--text-color);opacity:0.7;margin-top:6px">
      MAPE reduction overall<br>(14.5% → 11.2%)
    </div>
  </div>
  <div class="g-card" style="text-align:center">
    <div style="font-size:2rem;font-weight:700;color:#0D9488;line-height:1">11.7%</div>
    <div style="font-size:0.82rem;color:var(--text-color);opacity:0.7;margin-top:6px">
      Invariant improvement rate<br>across all product lines
    </div>
  </div>
  <div class="g-card" style="text-align:center">
    <div style="font-size:2rem;font-weight:700;color:#0D9488;line-height:1">€161k/yr</div>
    <div style="font-size:0.82rem;color:var(--text-color);opacity:0.7;margin-top:6px">
      Central cost estimate<br>at 1,000 components, €200/unit
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    sdiv()

    st.markdown("### What this does not solve yet")
    st.markdown("""
This model assumes substitutions are clean. One part replaces another at a single point
in time. In practice there are often transition periods where both the old and new
component are being ordered simultaneously. That overlap creates a demand-splitting
effect that adds its own layer of procurement complexity. Capturing that transition
period properly is the logical next step for this work, and it would likely increase
the estimated saving rather than reduce it.
""")

    st.markdown("""
<div style="background:rgba(251,191,36,0.07);
            border:1px dashed rgba(251,191,36,0.4);
            border-radius:10px;padding:1.75rem 1.5rem;margin:1.5rem 0">

  <div style="font-size:0.85rem;font-weight:600;color:var(--text-color);
              opacity:0.75;margin-bottom:1.1rem;text-align:center">
    Partial substitution — illustrative
  </div>

  <div style="display:flex;align-items:flex-end;gap:3px;
              height:72px;max-width:520px;margin:0 auto 1rem">
    <div style="flex:1;background:rgba(37,99,235,0.6);border-radius:3px 3px 0 0;height:65%"></div>
    <div style="flex:1;background:rgba(37,99,235,0.6);border-radius:3px 3px 0 0;height:58%"></div>
    <div style="flex:1;background:rgba(37,99,235,0.6);border-radius:3px 3px 0 0;height:72%"></div>
    <div style="flex:1;background:rgba(37,99,235,0.6);border-radius:3px 3px 0 0;height:61%"></div>
    <div style="flex:1;background:rgba(37,99,235,0.6);border-radius:3px 3px 0 0;height:67%"></div>
    <div style="flex:1;background:rgba(37,99,235,0.6);border-radius:3px 3px 0 0;height:63%"></div>
    <div style="flex:1;display:flex;flex-direction:column;justify-content:flex-end;gap:2px">
      <div style="background:rgba(13,148,136,0.65);border-radius:2px 2px 0 0;height:16px"></div>
      <div style="background:rgba(37,99,235,0.35);height:30px"></div>
    </div>
    <div style="flex:1;display:flex;flex-direction:column;justify-content:flex-end;gap:2px">
      <div style="background:rgba(13,148,136,0.65);border-radius:2px 2px 0 0;height:26px"></div>
      <div style="background:rgba(37,99,235,0.35);height:18px"></div>
    </div>
    <div style="flex:1;display:flex;flex-direction:column;justify-content:flex-end;gap:2px">
      <div style="background:rgba(13,148,136,0.65);border-radius:2px 2px 0 0;height:38px"></div>
      <div style="background:rgba(37,99,235,0.35);height:8px"></div>
    </div>
    <div style="flex:1;background:rgba(13,148,136,0.6);border-radius:3px 3px 0 0;height:64%"></div>
    <div style="flex:1;background:rgba(13,148,136,0.6);border-radius:3px 3px 0 0;height:59%"></div>
    <div style="flex:1;background:rgba(13,148,136,0.6);border-radius:3px 3px 0 0;height:68%"></div>
  </div>

  <div style="display:flex;justify-content:center;gap:1.5rem;
              font-size:0.78rem;color:var(--text-color);opacity:0.65">
    <span style="display:flex;align-items:center;gap:5px">
      <span style="width:11px;height:11px;background:rgba(37,99,235,0.6);
                   border-radius:2px;display:inline-block"></span>Old part
    </span>
    <span style="display:flex;align-items:center;gap:5px">
      <span style="width:11px;height:11px;
                   background:linear-gradient(90deg,rgba(37,99,235,0.4),rgba(13,148,136,0.6));
                   border-radius:2px;display:inline-block"></span>Overlap
    </span>
    <span style="display:flex;align-items:center;gap:5px">
      <span style="width:11px;height:11px;background:rgba(13,148,136,0.6);
                   border-radius:2px;display:inline-block"></span>New part
    </span>
  </div>

  <p style="font-size:0.81rem;color:var(--text-color);opacity:0.6;
            text-align:center;margin:0.9rem auto 0;max-width:460px;line-height:1.6">
    During the overlap window, both part numbers carry partial demand. A model that
    assumes a clean cutover will misread both series. Handling this correctly is out
    of scope for the current analysis but is the natural next step.
  </p>

</div>
""", unsafe_allow_html=True)

    sdiv()

    st.markdown("### Where this fits")
    st.markdown("""
Manufacturers who are already thinking about how analytics and AI can improve their
S&OP processes will recognise this kind of intervention. It does not require a new
system. It does not require replacing existing tools. It works with the data that is
already there, the BOM records, the part number history, the substitution logs, and
adds a layer of functional continuity that standard forecasting tools currently lack.
The barrier to implementation is lower than it might appear, and the place it fits in
the process is already well defined.
""")

    st.markdown("""
<div class="g-card" style="margin:1.25rem 0">
  <div style="font-size:0.9rem;font-weight:600;color:var(--text-color);
              opacity:0.85;margin-bottom:1rem">
    What implementation requires
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem">
    <div style="display:flex;gap:10px">
      <div style="color:#0D9488;flex-shrink:0;margin-top:2px">✓</div>
      <div style="font-size:0.87rem;color:var(--text-color);opacity:0.82;line-height:1.6">
        <strong>BOM records</strong><br>
        The component hierarchy already exists in every ERP system. No new data collection required.
      </div>
    </div>
    <div style="display:flex;gap:10px">
      <div style="color:#0D9488;flex-shrink:0;margin-top:2px">✓</div>
      <div style="font-size:0.87rem;color:var(--text-color);opacity:0.82;line-height:1.6">
        <strong>Part number history</strong><br>
        Historical demand by part number is standard ERP output, already used for forecasting.
      </div>
    </div>
    <div style="display:flex;gap:10px">
      <div style="color:#0D9488;flex-shrink:0;margin-top:2px">✓</div>
      <div style="font-size:0.87rem;color:var(--text-color);opacity:0.82;line-height:1.6">
        <strong>Substitution log</strong><br>
        Engineering change orders record substitution events. The lookup table already exists.
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
The one thing required that most ERP systems do not maintain by default is the
functional ID mapping: a link that connects successor part numbers to their predecessors
under a shared identity. That link is the contribution of this work. It takes existing
data and makes it available across part number transitions, rather than starting the
history over each time a substitution occurs.
""")

    sdiv()

    st.markdown("""
<p class="muted">
All data in this analysis is synthetic. Normet Group is referenced as a representative
example of the class of manufacturer this work addresses: a global industrial equipment
company with €482m in revenue, operations in 30-plus countries, and more than 14,000
machines delivered to customers worldwide. These figures are drawn from publicly
available company information. No proprietary operational data has been used.
</p>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTING
# ══════════════════════════════════════════════════════════════════════════════
if page == "1 · Problem statement":
    page_1()
elif page == "2 · Data overview":
    page_2()
elif page == "3 · Model comparison":
    page_3()
elif page == "4 · Cost simulation":
    page_4()
elif page == "5 · Conclusion":
    page_5()
