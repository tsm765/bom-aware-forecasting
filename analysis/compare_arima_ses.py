"""
BOM-Aware Demand Forecasting — Step 3c: ARIMA vs SES Comparison Chart
======================================================================
Methodology justification figure showing:
  Panel A — Fit rate: ARIMA vs SES across all part numbers
  Panel B — Where ARIMA does fit: MAPE comparison ARIMA vs SES (same 20 PNs)
  Panel C — Transition window coverage: who can actually forecast post-substitution
  Panel D — MAPE by months-since-substitution: SES accuracy profile across window

Output: plots/fig9_arima_vs_ses_comparison.png
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

warnings.filterwarnings("ignore")

DATA_DIR  = Path("data")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

# ── palette ────────────────────────────────────────────────────────────────────
C = {
    "arima"  : "#1B2A4A",   # dark navy   — ARIMA
    "ses"    : "#2563EB",   # blue        — SES
    "warn"   : "#DC2626",   # red         — failure / blind spot
    "ok"     : "#0D9488",   # teal        — success
    "amber"  : "#D97706",
    "lgrey"  : "#E5E7EB",
    "bg"     : "#F8FAFC",
    "text"   : "#1F2937",
}

plt.rcParams.update({
    "font.family"       : "DejaVu Sans",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.facecolor"    : C["bg"],
    "figure.facecolor"  : "white",
    "axes.titlesize"    : 11,
    "axes.titleweight"  : "bold",
    "axes.labelsize"    : 9.5,
    "xtick.labelsize"   : 9,
    "ytick.labelsize"   : 9,
    "axes.grid"         : True,
    "grid.color"        : C["lgrey"],
    "grid.linewidth"    : 0.7,
})

# ── load data ──────────────────────────────────────────────────────────────────
arima = pd.read_csv(DATA_DIR / "naive_forecasts.csv",  parse_dates=["forecast_month"])
ses   = pd.read_csv(DATA_DIR / "ses_forecasts.csv",    parse_dates=["forecast_month"])

n_pns_total = arima["part_number"].nunique()

# ── computed stats ─────────────────────────────────────────────────────────────
arima_fitted      = arima[arima["fit_status"] == "fitted"]["part_number"].unique()
ses_fitted        = ses["part_number"].unique()
n_arima_fitted    = len(arima_fitted)
n_ses_fitted      = len(ses_fitted)
n_arima_insuff    = n_pns_total - n_arima_fitted

# transition window
arima_trans = arima[arima["in_transition_window"] == True]
ses_trans   = ses[ses["in_transition_window"] == True]
n_trans_total     = len(ses_trans)  # 61 — ground truth from SES (covers all)
n_arima_can_fcst  = (arima_trans["fit_status"] == "fitted").sum()
n_ses_can_fcst    = len(ses_trans)

# MAPE on the 20 PNs ARIMA could fit — both models on same PNs
arima_fit_rows  = arima[(arima["fit_status"]=="fitted")]
ses_same_pns    = ses[ses["part_number"].isin(arima_fitted)]
arima_mape_overall = arima_fit_rows["pct_error"].mean()
ses_mape_overall   = ses_same_pns["pct_error"].mean()
ses_mape_all       = ses["pct_error"].mean()

# MAPE by months-since-sub for SES
ses_ms = (ses[ses["in_transition_window"]==True]
          .groupby("months_since_sub")["pct_error"].mean()
          .reset_index())

# ── figure layout ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 9))
fig.suptitle(
    "Methodology justification: why SES is the primary naive baseline",
    fontsize=14, fontweight="bold", y=1.01, color=C["text"]
)

gs = fig.add_gridspec(2, 2, hspace=0.52, wspace=0.38)
ax_a = fig.add_subplot(gs[0, 0])   # fit rate
ax_b = fig.add_subplot(gs[0, 1])   # MAPE head-to-head on same 20 PNs
ax_c = fig.add_subplot(gs[1, 0])   # transition window coverage
ax_d = fig.add_subplot(gs[1, 1])   # SES MAPE by months-since-sub

# ─────────────────────────────────────────────────────────────────────────────
# PANEL A — Fit rate comparison
# ─────────────────────────────────────────────────────────────────────────────
ax = ax_a
models  = ["ARIMA\n(auto-order)", "SES\n(primary baseline)"]
fitted  = [n_arima_fitted,   n_ses_fitted]
insuff  = [n_arima_insuff,   0]
x       = np.array([0, 1])
w       = 0.45

b1 = ax.bar(x, fitted, width=w, color=[C["ok"], C["ses"]],
            label="Fitted", zorder=3, alpha=0.88)
b2 = ax.bar(x, insuff, width=w, bottom=fitted,
            color=C["warn"], label="Insufficient history", zorder=3, alpha=0.72)

for bar, n in zip(b1, fitted):
    ax.text(bar.get_x() + bar.get_width()/2, n/2,
            str(n), ha="center", va="center",
            fontsize=12, fontweight="bold", color="white")

for bar, bot, n in zip(b2, fitted, insuff):
    if n > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bot + n/2,
                str(n), ha="center", va="center",
                fontsize=12, fontweight="bold", color="white")

# pct labels on top
pcts = [f"{n_arima_fitted/n_pns_total*100:.0f}% fit", "100% fit"]
tops = [n_pns_total, n_ses_fitted]
for xi, (top, pct) in enumerate(zip(tops, pcts)):
    col = C["warn"] if xi == 0 else C["ok"]
    ax.text(xi, top + 2, pct, ha="center", va="bottom",
            fontsize=9.5, fontweight="bold", color=col)

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.set_ylim(0, n_pns_total * 1.18)
ax.set_ylabel("Part numbers")
ax.set_title("A  |  Fit rate across all 125 part numbers", loc="left")
ax.legend(fontsize=8.5, loc="upper right")
ax.yaxis.set_major_locator(ticker.MultipleLocator(25))

# annotation
ax.annotate(
    f"ARIMA cannot fit {n_arima_insuff} of {n_pns_total}\npart numbers — all post-substitution",
    xy=(0, n_arima_fitted + n_arima_insuff / 2),
    xytext=(0.55, n_pns_total * 0.62),
    fontsize=8, color=C["warn"],
    arrowprops=dict(arrowstyle="-|>", color=C["warn"], lw=1.2),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["warn"], alpha=0.85)
)

# ─────────────────────────────────────────────────────────────────────────────
# PANEL B — MAPE head-to-head on the 20 PNs ARIMA fitted
# ─────────────────────────────────────────────────────────────────────────────
ax = ax_b

# per-PN MAPE for both models on the overlapping 20 PNs
arima_pn_mape = (arima_fit_rows.groupby("part_number")["pct_error"]
                 .mean().reset_index().rename(columns={"pct_error":"arima_mape"}))
ses_pn_mape   = (ses_same_pns.groupby("part_number")["pct_error"]
                 .mean().reset_index().rename(columns={"pct_error":"ses_mape"}))
merged = arima_pn_mape.merge(ses_pn_mape, on="part_number").sort_values("arima_mape")

idx = np.arange(len(merged))
w   = 0.35
ax.bar(idx - w/2, merged["arima_mape"], width=w, color=C["arima"], alpha=0.85,
       label=f"ARIMA  (mean {arima_mape_overall:.1f}%)", zorder=3)
ax.bar(idx + w/2, merged["ses_mape"],   width=w, color=C["ses"],   alpha=0.85,
       label=f"SES     (mean {ses_mape_overall:.1f}%)", zorder=3)

ax.axhline(arima_mape_overall, color=C["arima"], linewidth=1.3,
           linestyle="--", alpha=0.7)
ax.axhline(ses_mape_overall,   color=C["ses"],   linewidth=1.3,
           linestyle="--", alpha=0.7)

ax.set_xticks(idx)
ax.set_xticklabels([pn.split("-")[0]+"\n"+"-".join(pn.split("-")[1:])
                    for pn in merged["part_number"]],
                   fontsize=5.8, rotation=0)
ax.set_ylabel("MAPE (%)")
ax.set_title("B  |  MAPE on the 20 PNs ARIMA could fit\n(same part numbers, same test months)",
             loc="left")
ax.legend(fontsize=8.5)
ax.set_ylim(0, merged[["arima_mape","ses_mape"]].values.max() * 1.25)

# annotation
diff = arima_mape_overall - ses_mape_overall
sign = "+" if diff > 0 else ""
ax.text(len(merged)*0.62, arima_mape_overall * 1.12,
        f"ARIMA {sign}{diff:.1f}pp vs SES\non same series",
        fontsize=8, color=C["arima"],
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["arima"], alpha=0.8))

# ─────────────────────────────────────────────────────────────────────────────
# PANEL C — Transition window forecast coverage
# ─────────────────────────────────────────────────────────────────────────────
ax = ax_c

categories = ["ARIMA", "SES"]
can_forecast = [n_arima_can_fcst, n_ses_can_fcst]
cannot       = [n_trans_total - n_arima_can_fcst, 0]

b1 = ax.bar([0, 1], can_forecast, width=0.5,
            color=[C["ok"], C["ses"]], alpha=0.88, label="Has forecast", zorder=3)
b2 = ax.bar([0, 1], cannot, width=0.5, bottom=can_forecast,
            color=C["warn"], alpha=0.75, label="No forecast (blind spot)", zorder=3)

for bar, n in zip(b1, can_forecast):
    if n > 0:
        ax.text(bar.get_x() + bar.get_width()/2, n/2,
                str(n), ha="center", va="center",
                fontsize=14, fontweight="bold", color="white")
for bar, bot, n in zip(b2, can_forecast, cannot):
    if n > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bot + n/2,
                str(n), ha="center", va="center",
                fontsize=14, fontweight="bold", color="white")

for xi, (n_fc, n_tot) in enumerate(zip(can_forecast, [n_trans_total]*2)):
    pct = n_fc / n_tot * 100
    col = C["warn"] if pct < 50 else C["ok"]
    ax.text(xi, n_trans_total + 1.5, f"{pct:.0f}%",
            ha="center", fontsize=11, fontweight="bold", color=col)

ax.set_xticks([0, 1])
ax.set_xticklabels(["ARIMA\n(auto-order)", "SES\n(primary baseline)"], fontsize=10)
ax.set_ylim(0, n_trans_total * 1.2)
ax.set_ylabel("Forecast-month rows (0–6 months post-sub)")
ax.set_title("C  |  Transition-window forecast coverage\n(61 rows across 77 substitution events)",
             loc="left")
ax.legend(fontsize=8.5, loc="upper right")

# "these are real demand values" annotation
ax.annotate(
    f"61 real demand observations.\nARIMA produces no forecast\nfor any of them.",
    xy=(0, n_trans_total / 2),
    xytext=(0.42, n_trans_total * 0.72),
    fontsize=8, color=C["warn"],
    arrowprops=dict(arrowstyle="-|>", color=C["warn"], lw=1.2),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["warn"], alpha=0.85)
)

# ─────────────────────────────────────────────────────────────────────────────
# PANEL D — SES MAPE by months-since-substitution
# ─────────────────────────────────────────────────────────────────────────────
ax = ax_d

# counts per month for secondary axis
counts = (ses[ses["in_transition_window"]==True]
          .groupby("months_since_sub").size().reset_index(name="n"))
ms_vals  = ses_ms["months_since_sub"].values
mape_vals = ses_ms["pct_error"].values

bar_colors = [C["warn"] if m <= 2 else C["ses"] for m in ms_vals]
bars = ax.bar(ms_vals, mape_vals, color=bar_colors, alpha=0.80,
              width=0.6, zorder=3)

# overlay stable-period MAPE as reference line
stable_mape = ses[~ses["in_transition_window"]]["pct_error"].mean()
ax.axhline(stable_mape, color=C["amber"], linewidth=1.8, linestyle="--",
           label=f"Stable-period MAPE ({stable_mape:.1f}%)", zorder=4)

# value labels
for bar, val, m in zip(bars, mape_vals, ms_vals):
    ax.text(bar.get_x() + bar.get_width()/2,
            val + 0.4,
            f"{val:.1f}%", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold",
            color=C["warn"] if m <= 2 else C["text"])

# n labels at foot of bars
for m in ms_vals:
    n_row = counts[counts["months_since_sub"]==m]["n"].values
    if len(n_row):
        ax.text(m, 0.5, f"n={n_row[0]}",
                ha="center", va="bottom", fontsize=7.5, color="white",
                fontweight="bold")

ax.set_xlabel("Months since substitution event")
ax.set_ylabel("MAPE (%)")
ax.set_title("D  |  SES accuracy across post-substitution window\n"
             "(red = most acute period; amber = stable-period reference)",
             loc="left")
ax.set_xticks(ms_vals)
ax.set_xticklabels([f"+{int(m)}m" for m in ms_vals])
ax.set_ylim(0, max(mape_vals) * 1.35)
ax.legend(fontsize=8.5)

# shade first 2 months
if len(ms_vals) and ms_vals[0] <= 2:
    ax.axvspan(ms_vals[0] - 0.4, min(2, ms_vals[-1]) + 0.4,
               color=C["warn"], alpha=0.06, zorder=0,
               label="Acute window (0–2m)")

# footnote
fig.text(0.5, -0.025,
         "Note: ARIMA order selected by AIC across p∈{0–3}, d∈{0–1}, q∈{0–3}. "
         "SES α optimised by minimising one-step-ahead MSE on training series. "
         "Both models trained on same train/test split (last 3 months held out). "
         "MAPE computed as mean |forecast–actual| / max(actual,1) × 100.",
         ha="center", fontsize=8, color=C["text"], alpha=0.7,
         style="italic")

plt.tight_layout()
fig.savefig(PLOTS_DIR / "fig9_arima_vs_ses_comparison.png",
            dpi=150, bbox_inches="tight")
plt.close()

print("Saved: plots/fig9_arima_vs_ses_comparison.png")

# ── quick printed summary for record ──────────────────────────────────────────
print(f"\nKey numbers embedded in the figure:")
print(f"  ARIMA fit rate          : {n_arima_fitted}/{n_pns_total} ({n_arima_fitted/n_pns_total*100:.0f}%)")
print(f"  SES   fit rate          : {n_ses_fitted}/{n_pns_total} (100%)")
print(f"  Transition coverage — ARIMA : {n_arima_can_fcst}/{n_trans_total} (0%)")
print(f"  Transition coverage — SES   : {n_ses_can_fcst}/{n_trans_total} (100%)")
print(f"  MAPE on shared 20 PNs — ARIMA: {arima_mape_overall:.1f}%")
print(f"  MAPE on shared 20 PNs — SES  : {ses_mape_overall:.1f}%")
print(f"  SES MAPE all 125 PNs         : {ses_mape_all:.1f}%")
print(f"  SES transition-window MAPE   : {ses[ses['in_transition_window']==True]['pct_error'].mean():.1f}%")
print(f"  SES stable-period MAPE       : {stable_mape:.1f}%")
