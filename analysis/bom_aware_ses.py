"""
BOM-Aware Demand Forecasting — Step 4: BOM-Aware SES Model
===========================================================
Approach 2: SES fitted on the CHAINED demand history per functional ID.

The only difference from Step 3b (naive SES) is the input series:
  - Naive   : series = demand for THIS part number only (resets at substitution)
  - BOM-aware: series = demand chained across ALL part numbers sharing a
               functional ID, ordered chronologically

This gives the model 36 months of history regardless of when the current
part number was introduced — it sees the full demand trajectory of the
functional role, not the truncated history of the SKU.

After fitting on the chained series:
  - Forecast 3 months forward (months 34, 35, 36)
  - Map results back to the currently active part number (the last PN in chain)
  - Compute MAPE / MAE identically to Step 3b for direct comparison

Output schema (identical to ses_forecasts.csv)
----------------------------------------------
  part_number, functional_id, product_line,
  forecast_month, months_of_history,
  months_since_sub, in_transition_window,
  forecast, actual, abs_error, pct_error,
  fit_status, alpha, model_type

Files written
-------------
  data/bom_forecasts.csv
  data/bom_summary.csv
  data/bom_transition.csv

Comparison figures
------------------
  plots/fig10_bom_transition_mape.png   ← Panel D equivalent (matches fig9 panel D format)
  plots/fig11_naive_vs_bom_full.png     ← full side-by-side comparison
"""

import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import minimize_scalar

warnings.filterwarnings("ignore")

DATA_DIR          = Path("data")
PLOTS_DIR         = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

N_FORECAST        = 3
TRANSITION_WINDOW = 6
DEFAULT_ALPHA     = 0.5

# ── shared colour palette ──────────────────────────────────────────────────────
C = {
    "naive"  : "#2563EB",   # blue  — naive SES
    "bom"    : "#0D9488",   # teal  — BOM-aware SES
    "warn"   : "#DC2626",
    "ok"     : "#16A34A",
    "amber"  : "#D97706",
    "navy"   : "#1B2A4A",
    "lgrey"  : "#E5E7EB",
    "bg"     : "#F8FAFC",
    "text"   : "#1F2937",
    "pl"     : ["#2563EB","#0D9488","#D97706","#7C3AED"],
}

plt.rcParams.update({
    "font.family"     : "DejaVu Sans",
    "axes.spines.top" : False, "axes.spines.right": False,
    "axes.facecolor"  : C["bg"], "figure.facecolor": "white",
    "axes.titlesize"  : 11,     "axes.titleweight" : "bold",
    "axes.labelsize"  : 9.5,    "xtick.labelsize"  : 9,
    "ytick.labelsize" : 9,      "axes.grid"        : True,
    "grid.color"      : C["lgrey"], "grid.linewidth": 0.7,
})

# ══════════════════════════════════════════════════════════════════════════════
#  SES (identical implementation to Step 3b)
# ══════════════════════════════════════════════════════════════════════════════
def _ses_mse(alpha, y):
    s, sse = y[0], 0.0
    for t in range(1, len(y)):
        sse += (y[t] - s) ** 2
        s    = alpha * y[t] + (1 - alpha) * s
    return sse / max(len(y) - 1, 1)

def ses_fit(y_train):
    n = len(y_train)
    if n == 1:
        return dict(alpha=DEFAULT_ALPHA, level=float(y_train[0]), mse=np.nan)
    result = minimize_scalar(_ses_mse, args=(y_train,),
                             bounds=(0.05, 0.95), method="bounded",
                             options={"xatol": 1e-6})
    s = y_train[0]
    for t in range(1, n):
        s = result.x * y_train[t] + (1 - result.x) * s
    return dict(alpha=result.x, level=s, mse=result.fun)

def ses_forecast(model, steps):
    return np.clip(np.round(np.full(steps, model["level"])), 0, None)

# ══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*64)
print("  BOM-Aware SES — chained functional-ID series (Step 4)")
print("="*64)

df  = pd.read_csv(DATA_DIR / "demand_history.csv",       parse_dates=["date"])
lkp = pd.read_csv(DATA_DIR / "functional_id_lookup.csv", parse_dates=["active_from"])
sub = pd.read_csv(DATA_DIR / "substitution_events.csv",  parse_dates=["event_date"])

sub_introduced = set(sub["new_part_number"].unique())
lkp["active_from"] = pd.to_datetime(lkp["active_from"])
pn_active_from = lkp.set_index("part_number")["active_from"].to_dict()

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP — one SES per FUNCTIONAL FAMILY (chained series)
# ══════════════════════════════════════════════════════════════════════════════
families = df.groupby(["product_line","functional_id"])
n_families = len(families)
results = []

print(f"\nFitting BOM-aware SES for {n_families} functional families …")
t_start = time.time()

for (pl, fid), fam_df in families:
    # ── build the chained series ────────────────────────────────────────────
    # Sort ALL part numbers in this family chronologically — this is the chain.
    # The demand_history already has one row per (date, part_number) with no
    # overlaps, so a simple date sort gives the continuous functional series.
    chain = fam_df.sort_values("date").reset_index(drop=True)
    n_chain = len(chain)                      # always 36 for our dataset

    y_chain    = chain["demand"].values.astype(float)
    n_train    = n_chain - N_FORECAST
    y_train    = y_chain[:n_train]            # 33 months — full functional history
    y_test     = y_chain[n_train:]            # 3 held-out months
    test_dates = chain["date"].values[n_train:]
    test_pns   = chain["part_number"].values[n_train:]   # active PN each test month

    # ── fit on chained history ──────────────────────────────────────────────
    model     = ses_fit(y_train)
    forecasts = ses_forecast(model, N_FORECAST)

    # ── store rows — mapped back to active part number ──────────────────────
    for i, (td, act, fcast, pn) in enumerate(
            zip(test_dates, y_test, forecasts, test_pns)):
        td_ts = pd.Timestamp(td)

        is_sub = pn in sub_introduced
        active_from = pn_active_from.get(pn, None)
        months_since_sub = None
        in_window        = False
        if is_sub and active_from is not None:
            ms = (td_ts.year - active_from.year)*12 + \
                 (td_ts.month - active_from.month)
            months_since_sub = ms
            in_window = 0 <= ms <= TRANSITION_WINDOW

        abs_err = abs(fcast - act)
        pct_err = abs_err / max(act, 1) * 100

        results.append(dict(
            part_number=pn, functional_id=fid, product_line=pl,
            forecast_month=td_ts,
            months_of_history=int(n_train),    # always 33 — full chain
            months_since_sub=months_since_sub,
            in_transition_window=in_window,
            forecast=fcast, actual=int(act),
            abs_error=abs_err, pct_error=pct_err,
            fit_status="fitted",
            alpha=round(model["alpha"], 4),
            model_type="BOM-aware SES",
        ))

elapsed = time.time() - t_start
print(f"  Done in {elapsed:.3f}s — all {n_families} families fitted on 33-month chains.\n")

# ══════════════════════════════════════════════════════════════════════════════
#  ASSEMBLE
# ══════════════════════════════════════════════════════════════════════════════
res_df = pd.DataFrame(results)
res_df["forecast_month"] = pd.to_datetime(res_df["forecast_month"])

# deduplicate: a PN may appear in multiple families' test windows if it spans
# the boundary — take its row from its own family (already guaranteed by loop)
n_pns_covered = res_df["part_number"].nunique()

# ── summary per family ─────────────────────────────────────────────────────
summary_rows = []
for (pl, fid), grp in res_df.groupby(["product_line","functional_id"]):
    t = grp[grp["in_transition_window"]==True]
    summary_rows.append(dict(
        product_line=pl, functional_id=fid,
        n_forecasts=len(grp),
        mape=round(grp["pct_error"].mean(), 2),
        mae=round(grp["abs_error"].mean(), 2),
        n_transition_rows=len(t),
        transition_mape=round(t["pct_error"].mean(), 2) if len(t) else np.nan,
        transition_mae=round(t["abs_error"].mean(), 2)  if len(t) else np.nan,
    ))
summary_df    = pd.DataFrame(summary_rows)
transition_df = res_df[res_df["in_transition_window"]==True].copy()

# ══════════════════════════════════════════════════════════════════════════════
#  PRINTED REPORT
# ══════════════════════════════════════════════════════════════════════════════
sep = "─" * 64
print("="*64)
print("  BOM-AWARE SES RESULTS")
print("="*64)

n_pns_total = df["part_number"].nunique()
print(f"\n{sep}")
print("  Input series construction")
print(sep)
print(f"  Functional families fitted   : {n_families}")
print(f"  Training series length       : 33 months (full chained history)")
print(f"  Part numbers in test window  : {n_pns_covered}")
print(f"  Fit rate                     : 100%  (33 obs >> minimum for SES)")

# alpha comparison
ses_df = pd.read_csv(DATA_DIR / "ses_forecasts.csv")
bom_alpha = res_df.drop_duplicates("functional_id")["alpha"]
ses_alpha = ses_df.drop_duplicates("part_number")["alpha"]
print(f"\n  α (smoothing param) — BOM-aware  mean={bom_alpha.mean():.3f}  "
      f"median={bom_alpha.median():.3f}")
print(f"  α (smoothing param) — Naive SES  mean={ses_alpha.mean():.3f}  "
      f"median={ses_alpha.median():.3f}")
print(f"  Lower α in BOM-aware model = more weight on history (longer series).")

print(f"\n{sep}")
print("  Forecast accuracy")
print(sep)
bom_mape  = res_df["pct_error"].mean()
bom_mae   = res_df["abs_error"].mean()
ses_all   = ses_df["pct_error"].mean()
ses_mae_all = ses_df["abs_error"].mean()
print(f"  BOM-aware MAPE : {bom_mape:.1f}%   (Naive SES: {ses_all:.1f}%)  "
      f"Δ = {bom_mape - ses_all:+.1f}pp")
print(f"  BOM-aware MAE  : {bom_mae:.1f}    (Naive SES: {ses_mae_all:.1f})  "
      f"Δ = {bom_mae - ses_mae_all:+.1f}")
print(f"\n  By product line:")
for pl in sorted(res_df["product_line"].unique()):
    b = res_df[res_df["product_line"]==pl]
    s = ses_df[ses_df["product_line"]==pl]
    print(f"    {pl:<12}  BOM MAPE={b['pct_error'].mean():.1f}%  "
          f"Naive={s['pct_error'].mean():.1f}%  "
          f"Δ={b['pct_error'].mean()-s['pct_error'].mean():+.1f}pp")

print(f"\n{sep}")
print("  Transition window analysis (0–6 months post-substitution)")
print(sep)
t_bom   = transition_df
ses_trans_df = pd.read_csv(DATA_DIR / "ses_transition.csv")
t_ses   = ses_trans_df

print(f"  Transition rows              : {len(t_bom)}  (same 61 events)")
print(f"  BOM-aware transition MAPE    : {t_bom['pct_error'].mean():.1f}%")
print(f"  Naive SES  transition MAPE   : {t_ses['pct_error'].mean():.1f}%")
print(f"  Improvement                  : "
      f"{t_ses['pct_error'].mean() - t_bom['pct_error'].mean():+.1f}pp")
print(f"\n  BOM-aware transition MAE     : {t_bom['abs_error'].mean():.1f}")
print(f"  Naive SES  transition MAE    : {t_ses['abs_error'].mean():.1f}")
print(f"  Improvement                  : "
      f"{t_ses['abs_error'].mean() - t_bom['abs_error'].mean():+.1f} units/month")

print(f"\n  Accuracy by months-since-substitution:")
print(f"  {'Month':<9} {'BOM MAPE':>10} {'Naive MAPE':>12} {'Δ':>8} "
      f"{'BOM MAE':>10} {'Naive MAE':>11} {'Δ MAE':>8}")
print(f"  {'-'*69}")
for ms in range(TRANSITION_WINDOW + 1):
    b_rows = t_bom[t_bom["months_since_sub"]==ms]
    s_rows = t_ses[t_ses["months_since_sub"]==ms]
    if len(b_rows) == 0 and len(s_rows) == 0:
        continue
    b_mape = b_rows["pct_error"].mean() if len(b_rows) else np.nan
    s_mape = s_rows["pct_error"].mean() if len(s_rows) else np.nan
    b_mae  = b_rows["abs_error"].mean() if len(b_rows) else np.nan
    s_mae  = s_rows["abs_error"].mean() if len(s_rows) else np.nan
    d_mape = b_mape - s_mape
    d_mae  = b_mae - s_mae
    n = len(b_rows)
    print(f"  +{ms}m  (n={n:2d})  {b_mape:>9.1f}%  {s_mape:>11.1f}%  "
          f"{d_mape:>+7.1f}pp  {b_mae:>9.1f}  {s_mae:>10.1f}  {d_mae:>+7.1f}")

# stable period
bom_stable  = res_df[~res_df["in_transition_window"]]["pct_error"].mean()
ses_stable  = ses_df[~ses_df["in_transition_window"]]["pct_error"].mean()
print(f"\n  Stable-period MAPE — BOM-aware: {bom_stable:.1f}%  "
      f"Naive: {ses_stable:.1f}%  Δ={bom_stable-ses_stable:+.1f}pp")

print(f"\n{'='*64}\n")

# ══════════════════════════════════════════════════════════════════════════════
#  SAVE DATA OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
res_df.to_csv(DATA_DIR / "bom_forecasts.csv", index=False)
summary_df.to_csv(DATA_DIR / "bom_summary.csv", index=False)
transition_df.to_csv(DATA_DIR / "bom_transition.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
#  FIG 10 — Panel D equivalent: BOM-aware MAPE by months-since-sub
#            Same format as fig9 panel D for direct side-by-side comparison
# ══════════════════════════════════════════════════════════════════════════════
def plot_transition_panel(ax, trans_data, stable_mape, color, title_label, model_label):
    """Reusable panel-D style plot."""
    ms_mape = (trans_data.groupby("months_since_sub")["pct_error"]
               .mean().reset_index())
    ms_n    = (trans_data.groupby("months_since_sub").size()
               .reset_index(name="n"))
    ms_vals  = ms_mape["months_since_sub"].values
    mape_val = ms_mape["pct_error"].values

    bar_colors = [C["warn"] if m <= 2 else color for m in ms_vals]
    bars = ax.bar(ms_vals, mape_val, color=bar_colors, alpha=0.82,
                  width=0.6, zorder=3)

    ax.axhline(stable_mape, color=C["amber"], linewidth=1.8, linestyle="--",
               label=f"Stable-period MAPE ({stable_mape:.1f}%)", zorder=4)

    for bar, val, m in zip(bars, mape_val, ms_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=9, fontweight="bold",
                color=C["warn"] if m <= 2 else C["text"])

    for m in ms_vals:
        n_row = ms_n[ms_n["months_since_sub"]==m]["n"].values
        if len(n_row):
            ax.text(m, 0.4, f"n={n_row[0]}",
                    ha="center", va="bottom", fontsize=7.5,
                    color="white", fontweight="bold")

    ax.set_xlabel("Months since substitution event")
    ax.set_ylabel("MAPE (%)")
    ax.set_title(title_label, loc="left")
    ax.set_xticks(ms_vals)
    ax.set_xticklabels([f"+{int(m)}m" for m in ms_vals])
    ax.legend(fontsize=8.5)
    return ms_vals, mape_val

# ── standalone fig10: BOM-aware panel D ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
bom_stable_mape = res_df[~res_df["in_transition_window"]]["pct_error"].mean()
plot_transition_panel(
    ax, transition_df, bom_stable_mape,
    C["bom"],
    "BOM-aware SES — MAPE across post-substitution window\n"
    "(teal = BOM-aware; amber = stable-period reference)",
    "BOM-aware SES"
)
y_max = transition_df.groupby("months_since_sub")["pct_error"].mean().max()
ax.set_ylim(0, y_max * 1.45)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "fig10_bom_transition_mape.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → fig10_bom_transition_mape.png")

# ══════════════════════════════════════════════════════════════════════════════
#  FIG 11 — Full side-by-side: Naive SES vs BOM-aware SES (4-panel)
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(15, 10))
fig.suptitle(
    "Naive SES vs BOM-Aware SES — full comparison",
    fontsize=14, fontweight="bold", y=1.01, color=C["text"]
)
gs = fig.add_gridspec(2, 2, hspace=0.52, wspace=0.38)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])

# ── Panel A: overall MAPE comparison by product line ──────────────────────────
ax = ax_a
pls = sorted(res_df["product_line"].unique())
x   = np.arange(len(pls))
w   = 0.35
ses_mapes = [ses_df[ses_df["product_line"]==pl]["pct_error"].mean() for pl in pls]
bom_mapes = [res_df[res_df["product_line"]==pl]["pct_error"].mean() for pl in pls]

b1 = ax.bar(x - w/2, ses_mapes, width=w, color=C["naive"], alpha=0.85,
            label=f"Naive SES  (mean {ses_all:.1f}%)", zorder=3)
b2 = ax.bar(x + w/2, bom_mapes, width=w, color=C["bom"],   alpha=0.85,
            label=f"BOM-aware  (mean {bom_mape:.1f}%)", zorder=3)

for bars, vals in [(b1, ses_mapes), (b2, bom_mapes)]:
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.2,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([pl.replace("PL-","") for pl in pls], fontsize=10)
ax.set_ylabel("MAPE (%)")
ax.set_title("A  |  Overall MAPE by product line", loc="left")
ax.legend(fontsize=9)
ax.set_ylim(0, max(max(ses_mapes), max(bom_mapes)) * 1.3)
ax.axhline(ses_all, color=C["naive"], linewidth=1.2, linestyle=":", alpha=0.5)
ax.axhline(bom_mape, color=C["bom"], linewidth=1.2, linestyle=":", alpha=0.5)

# ── Panel B: per-family MAPE scatter — Naive vs BOM-aware ─────────────────────
ax = ax_b
ses_fam = (ses_df.groupby(["product_line","functional_id"])["pct_error"]
           .mean().reset_index().rename(columns={"pct_error":"ses_mape"}))
bom_fam = (res_df.groupby(["product_line","functional_id"])["pct_error"]
           .mean().reset_index().rename(columns={"pct_error":"bom_mape"}))
fam_merged = ses_fam.merge(bom_fam, on=["product_line","functional_id"])
pl_colors_map = dict(zip(pls, C["pl"]))

for pl in pls:
    sub_m = fam_merged[fam_merged["product_line"]==pl]
    ax.scatter(sub_m["ses_mape"], sub_m["bom_mape"],
               color=pl_colors_map[pl], alpha=0.75, s=55,
               label=pl.replace("PL-",""), zorder=3)

lim = max(fam_merged[["ses_mape","bom_mape"]].values.max(), 1) * 1.1
ax.plot([0, lim], [0, lim], color=C["navy"], linewidth=1.2,
        linestyle="--", alpha=0.5, label="No change", zorder=1)

n_improved = (fam_merged["bom_mape"] < fam_merged["ses_mape"]).sum()
n_total_f  = len(fam_merged)
ax.text(lim * 0.65, lim * 0.08,
        f"{n_improved}/{n_total_f} families\nimproved",
        fontsize=9, color=C["bom"], fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["bom"], alpha=0.85))

ax.set_xlabel("Naive SES MAPE (%) — per family")
ax.set_ylabel("BOM-aware SES MAPE (%) — per family")
ax.set_title("B  |  Per-family MAPE: Naive vs BOM-aware\n(below diagonal = BOM-aware wins)",
             loc="left")
ax.legend(fontsize=8, ncol=2)
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)

# shade "BOM wins" region
ax.fill_between([0, lim], [0, lim], [0, 0], alpha=0.04, color=C["bom"])
ax.text(lim*0.08, lim*0.02, "BOM-aware wins →", fontsize=7.5,
        color=C["bom"], alpha=0.7)

# ── Panel C: transition window MAPE side-by-side ──────────────────────────────
ax = ax_c
ms_vals = sorted(set(transition_df["months_since_sub"].dropna().unique()) |
                 set(ses_trans_df["months_since_sub"].dropna().unique()))
ms_vals = [m for m in ms_vals if not np.isnan(m)]

x2  = np.arange(len(ms_vals))
w2  = 0.35

ses_t_mape = [ses_trans_df[ses_trans_df["months_since_sub"]==m]["pct_error"].mean()
              if len(ses_trans_df[ses_trans_df["months_since_sub"]==m]) else np.nan
              for m in ms_vals]
bom_t_mape = [transition_df[transition_df["months_since_sub"]==m]["pct_error"].mean()
              if len(transition_df[transition_df["months_since_sub"]==m]) else np.nan
              for m in ms_vals]
ns          = [len(transition_df[transition_df["months_since_sub"]==m]) for m in ms_vals]

b1 = ax.bar(x2 - w2/2, ses_t_mape, width=w2, color=C["naive"], alpha=0.85,
            label=f"Naive SES  ({ses_trans_df['pct_error'].mean():.1f}% avg)", zorder=3)
b2 = ax.bar(x2 + w2/2, bom_t_mape, width=w2, color=C["bom"],   alpha=0.85,
            label=f"BOM-aware  ({transition_df['pct_error'].mean():.1f}% avg)", zorder=3)

for bars, vals in [(b1, ses_t_mape), (b2, bom_t_mape)]:
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.3,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

for xi, (m, n) in enumerate(zip(ms_vals, ns)):
    ax.text(xi, 0.4, f"n={n}", ha="center", va="bottom",
            fontsize=7, color=C["text"], alpha=0.7)

ax.set_xticks(x2)
ax.set_xticklabels([f"+{int(m)}m" for m in ms_vals])
ax.set_xlabel("Months since substitution event")
ax.set_ylabel("MAPE (%)")
ax.set_title("C  |  Transition-window MAPE by month\n(0–6 months post-substitution)",
             loc="left")
ax.legend(fontsize=9)
ax.set_ylim(0, max([v for v in ses_t_mape + bom_t_mape
                    if not np.isnan(v)]) * 1.35)

# ── Panel D: MAE comparison across transition window ──────────────────────────
ax = ax_d
ses_t_mae = [ses_trans_df[ses_trans_df["months_since_sub"]==m]["abs_error"].mean()
             if len(ses_trans_df[ses_trans_df["months_since_sub"]==m]) else np.nan
             for m in ms_vals]
bom_t_mae = [transition_df[transition_df["months_since_sub"]==m]["abs_error"].mean()
             if len(transition_df[transition_df["months_since_sub"]==m]) else np.nan
             for m in ms_vals]

b1 = ax.bar(x2 - w2/2, ses_t_mae, width=w2, color=C["naive"], alpha=0.85,
            label=f"Naive SES  ({ses_trans_df['abs_error'].mean():.1f} avg)", zorder=3)
b2 = ax.bar(x2 + w2/2, bom_t_mae, width=w2, color=C["bom"],   alpha=0.85,
            label=f"BOM-aware  ({transition_df['abs_error'].mean():.1f} avg)", zorder=3)

for bars, vals in [(b1, ses_t_mae), (b2, bom_t_mae)]:
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.15,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

# improvement arrows
for xi, (s, b) in enumerate(zip(ses_t_mae, bom_t_mae)):
    if not np.isnan(s) and not np.isnan(b) and b < s:
        mid = xi + w2/2
        ax.annotate("", xy=(mid, b + 0.3), xytext=(mid, s - 0.3),
                    arrowprops=dict(arrowstyle="-|>", color=C["ok"],
                                   lw=1.5, mutation_scale=10))

ax.set_xticks(x2)
ax.set_xticklabels([f"+{int(m)}m" for m in ms_vals])
ax.set_xlabel("Months since substitution event")
ax.set_ylabel("MAE (units / month)")
ax.set_title("D  |  Transition-window MAE by month\n(arrows = BOM-aware improvement)",
             loc="left")
ax.legend(fontsize=9)
ax.set_ylim(0, max([v for v in ses_t_mae + bom_t_mae
                    if not np.isnan(v)]) * 1.35)

# footnote
fig.text(0.5, -0.022,
         "Both models use identical SES implementation (α optimised by MSE). "
         "Naive: series = per-part-number history. "
         "BOM-aware: series = chained functional-ID history (33 months). "
         "Test set: last 3 months of each series.",
         ha="center", fontsize=8, color=C["text"], alpha=0.7, style="italic")

plt.tight_layout()
fig.savefig(PLOTS_DIR / "fig11_naive_vs_bom_full.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → fig11_naive_vs_bom_full.png")

# ── save data ──────────────────────────────────────────────────────────────────
print(f"\nFiles written:")
for f in ["bom_forecasts.csv","bom_summary.csv","bom_transition.csv"]:
    size = (DATA_DIR / f).stat().st_size / 1024
    print(f"  data/{f:<35} ({size:.1f} KB)")
print("\nBOM-aware SES complete. ✓")
