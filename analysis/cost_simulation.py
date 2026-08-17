"""
BOM-Aware Demand Forecasting — Step 6: Cost Simulation
=======================================================
Converts MAE forecast error differential into annual inventory cost in euros.

Three cost components, each using a confirmed parameter:
  1. Excess safety stock holding cost  (uses: Z, MAE, LT, unit_cost, holding_rate)
  2. Emergency ordering cost           (uses: ordering_cost, MAE ratio)
  3. Stockout exposure cost            (uses: Z service level, MAE, LT, unit_cost)

All parameters and their rationale are printed in an explicit table.
Sensitivity analysis across unit_cost €100–€500.

Outputs
-------
  data/cost_simulation.csv           ← full results at all unit cost levels
  plots/fig12_cost_sensitivity.png   ← annual saving vs unit cost (sensitivity chart)
  plots/fig13_cost_breakdown.png     ← stacked bar: 3 cost components at 3 price points
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

# ── colour palette ─────────────────────────────────────────────────────────────
C = {
    "naive"  : "#2563EB",
    "bom"    : "#0D9488",
    "comp1"  : "#1B2A4A",   # SS holding
    "comp2"  : "#D97706",   # emergency ordering
    "comp3"  : "#DC2626",   # stockout
    "saving" : "#16A34A",   # net saving
    "lgrey"  : "#E5E7EB",
    "bg"     : "#F8FAFC",
    "text"   : "#1F2937",
    "amber"  : "#D97706",
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
#  CONFIRMED PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
PARAMS = {
    # ── model inputs ──────────────────────────────────────────────────────────
    "mae_naive"         : 12.84,    # units/month — naive SES transition-window MAE
    "mae_bom"           : 11.35,    # units/month — BOM-aware SES transition-window MAE
    "n_subs_per_year"   : 26,       # events/year — 77 events / 36 months × 12
    # ── safety stock parameters ───────────────────────────────────────────────
    "Z"                 : 1.65,     # service level Z-score — 95% cycle service level
    "lead_time_weeks"   : 6,        # weeks — confirmed lead time
    # ── cost parameters ───────────────────────────────────────────────────────
    "unit_cost_base"    : 200,      # EUR — baseline unit cost
    "unit_cost_low"     : 100,      # EUR — sensitivity lower bound
    "unit_cost_high"    : 500,      # EUR — sensitivity upper bound
    "holding_rate"      : 0.22,     # annual fraction — industry standard 20–25%
    "ordering_cost"     : 100,      # EUR/PO — cost per purchase order
}

# derived
PARAMS["lead_time_months"] = PARAMS["lead_time_weeks"] / (52 / 12)   # = 1.3846 months
PARAMS["delta_mae"]        = PARAMS["mae_naive"] - PARAMS["mae_bom"]  # = 1.49
PARAMS["service_level"]    = 0.95
PARAMS["p_stockout"]       = 1 - PARAMS["service_level"]               # = 0.05

sep = "─" * 72

# ══════════════════════════════════════════════════════════════════════════════
#  PRINT PARAMETER TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("  Cost Simulation — Parameter Table")
print("="*72)

param_table = [
    ("Naive transition-window MAE",  f"{PARAMS['mae_naive']:.2f} units/month",
     "Direct output from ses_transition.csv"),
    ("BOM-aware transition-window MAE", f"{PARAMS['mae_bom']:.2f} units/month",
     "Direct output from bom_transition.csv"),
    ("ΔMAE (naive − BOM-aware)",     f"{PARAMS['delta_mae']:.2f} units/month",
     "The forecast error improvement being costed"),
    ("Substitution events / year",   f"{PARAMS['n_subs_per_year']} events",
     "77 events over 36 months → 77/3 = 25.7, rounded to 26"),
    ("Service level Z-score",        f"Z = {PARAMS['Z']} (95%)",
     "Standard for industrial spare parts; 95% cycle service level"),
    ("Lead time",                    f"{PARAMS['lead_time_weeks']} weeks "
                                     f"({PARAMS['lead_time_months']:.4f} months)",
     "Confirmed; converted: LT_months = LT_weeks / (52/12)"),
    ("σ_demand proxy",               "= MAE directly",
     "MAE ≈ σ for symmetric error distributions at this scale"),
    ("Unit cost — baseline",         f"€{PARAMS['unit_cost_base']}",
     "Representative mid-range component cost; sensitivity at €100 / €500"),
    ("Holding cost rate",            f"{PARAMS['holding_rate']*100:.0f}% / year",
     "Industry standard range 20–25%; covers capital, storage, obsolescence"),
    ("Ordering cost / PO",           f"€{PARAMS['ordering_cost']}",
     "Internal cost per purchase order (admin, approval, logistics)"),
]

print(f"\n  {'Parameter':<38} {'Value':<28} {'Rationale'}")
print(f"  {'-'*38} {'-'*28} {'-'*40}")
for name, val, rationale in param_table:
    print(f"  {name:<38} {val:<28} {rationale}")

# ══════════════════════════════════════════════════════════════════════════════
#  COST MODEL — THREE COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("  Cost Model — Three Components")
print("="*72)

print(f"""
  Component 1 — Excess Safety Stock Holding Cost
  ─────────────────────────────────────────────────
  The naive model's higher σ requires more safety stock to achieve the
  same 95% service level. The planner holds ΔSS extra units per event,
  incurring holding cost on that buffer for the full year.

    SS  = Z × σ_demand × √(LT_months)
    ΔSS = Z × ΔMAE × √(LT_months)
        = {PARAMS['Z']} × {PARAMS['delta_mae']:.2f} × √{PARAMS['lead_time_months']:.4f}
        = {PARAMS['Z']} × {PARAMS['delta_mae']:.2f} × {np.sqrt(PARAMS['lead_time_months']):.4f}
        = {PARAMS['Z'] * PARAMS['delta_mae'] * np.sqrt(PARAMS['lead_time_months']):.3f} units / event

    Annual holding cost saving = ΔSS × unit_cost × holding_rate × n_events

  Component 2 — Emergency Ordering Cost
  ─────────────────────────────────────────────────
  The naive model's higher forecast error causes proportionally more
  emergency reorders during the transition window. Extra POs per event
  scale with the relative MAE improvement.

    Extra POs / event = (MAE_naive / MAE_bom − 1)
                      = ({PARAMS['mae_naive']:.2f} / {PARAMS['mae_bom']:.2f} − 1)
                      = {PARAMS['mae_naive']/PARAMS['mae_bom'] - 1:.4f} extra POs
    Annual cost saving = extra_POs × ordering_cost × n_events
                       = {PARAMS['mae_naive']/PARAMS['mae_bom'] - 1:.4f} × €{PARAMS['ordering_cost']} × {PARAMS['n_subs_per_year']}
                       = €{(PARAMS['mae_naive']/PARAMS['mae_bom'] - 1) * PARAMS['ordering_cost'] * PARAMS['n_subs_per_year']:.0f}  (unit-cost independent)

  Component 3 — Stockout Exposure Cost
  ─────────────────────────────────────────────────
  Given Z = {PARAMS['Z']}, P(stockout per cycle) = {PARAMS['p_stockout']:.2f}. When a stockout
  occurs, expected units short = ΔMAE × LT_months. The naive model's
  larger error pool generates proportionally more expected shortage units.

    Expected extra shortage units / event = ΔMAE × P(stockout) × LT_months
        = {PARAMS['delta_mae']:.2f} × {PARAMS['p_stockout']:.2f} × {PARAMS['lead_time_months']:.4f}
        = {PARAMS['delta_mae'] * PARAMS['p_stockout'] * PARAMS['lead_time_months']:.4f} units / event

    Stockout penalty per unit = unit_cost (emergency procurement at full cost)
    Annual stockout saving    = extra_shortage × unit_cost × n_events
""")

# ══════════════════════════════════════════════════════════════════════════════
#  COMPUTE COSTS ACROSS UNIT COST RANGE
# ══════════════════════════════════════════════════════════════════════════════
Z   = PARAMS["Z"]
de  = PARAMS["delta_mae"]
LT  = PARAMS["lead_time_months"]
hr  = PARAMS["holding_rate"]
oc  = PARAMS["ordering_cost"]
n   = PARAMS["n_subs_per_year"]
ps  = PARAMS["p_stockout"]

def compute_savings(unit_cost):
    delta_SS = Z * de * np.sqrt(LT)
    c1 = delta_SS * unit_cost * hr * n
    extra_pos_per_event = PARAMS["mae_naive"] / PARAMS["mae_bom"] - 1
    c2 = extra_pos_per_event * oc * n          # unit-cost independent
    extra_shortage = de * ps * LT * n
    c3 = extra_shortage * unit_cost
    total = c1 + c2 + c3
    return dict(unit_cost=unit_cost, c1_holding=c1, c2_ordering=c2,
                c3_stockout=c3, total_saving=total,
                delta_ss=delta_SS)

# spot compute for printed table
spot_costs = [compute_savings(uc) for uc in [100, 200, 500]]

print(f"{'='*72}")
print("  Annual Cost Savings: Naive SES → BOM-Aware SES")
print("="*72)
print(f"\n  {'Unit Cost':<12} {'SS Holding':>14} {'Emergency PO':>14} "
      f"{'Stockout':>12} {'TOTAL SAVING':>14}")
print(f"  {'-'*12} {'-'*14} {'-'*14} {'-'*12} {'-'*14}")
for r in spot_costs:
    print(f"  €{r['unit_cost']:<10,.0f} "
          f"  €{r['c1_holding']:>11,.0f}   "
          f"€{r['c2_ordering']:>11,.0f}   "
          f"€{r['c3_stockout']:>9,.0f}   "
          f"€{r['total_saving']:>11,.0f}")

print(f"\n  Notes:")
print(f"    · Emergency PO saving is fixed at "
      f"€{spot_costs[0]['c2_ordering']:.0f}/yr (independent of unit cost)")
print(f"    · ΔSS = {spot_costs[0]['delta_ss']:.2f} units per substitution event")
print(f"    · At €200/unit, SS holding drives "
      f"{spot_costs[1]['c1_holding']/spot_costs[1]['total_saving']*100:.0f}% of total saving")

# ── also show ABSOLUTE costs (naive vs BOM-aware), not just differential ───────
print(f"\n{sep}")
print("  Absolute annual inventory cost — Naive vs BOM-Aware (€200 baseline)")
print(sep)

uc = 200
ss_naive = Z * PARAMS["mae_naive"] * np.sqrt(LT)
ss_bom   = Z * PARAMS["mae_bom"]   * np.sqrt(LT)
holding_naive = ss_naive * uc * hr * n
holding_bom   = ss_bom   * uc * hr * n
eo_naive = 1 * oc * n                           # 1 PO per event baseline for naive
eo_bom   = (PARAMS["mae_bom"]/PARAMS["mae_naive"]) * oc * n
so_naive = PARAMS["mae_naive"] * ps * LT * uc * n
so_bom   = PARAMS["mae_bom"]   * ps * LT * uc * n
total_naive = holding_naive + eo_naive + so_naive
total_bom   = holding_bom   + eo_bom   + so_bom

print(f"\n  {'Component':<30} {'Naive SES':>12} {'BOM-Aware':>12} {'Saving':>12}")
print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12}")
print(f"  {'SS Holding Cost':<30} €{holding_naive:>10,.0f} €{holding_bom:>10,.0f} "
      f"€{holding_naive-holding_bom:>10,.0f}")
print(f"  {'Emergency Ordering Cost':<30} €{eo_naive:>10,.0f} €{eo_bom:>10,.0f} "
      f"€{eo_naive-eo_bom:>10,.0f}")
print(f"  {'Stockout Exposure Cost':<30} €{so_naive:>10,.0f} €{so_bom:>10,.0f} "
      f"€{so_naive-so_bom:>10,.0f}")
print(f"  {'─'*66}")
print(f"  {'TOTAL ANNUAL COST':<30} €{total_naive:>10,.0f} €{total_bom:>10,.0f} "
      f"€{total_naive-total_bom:>10,.0f}")
print(f"  {'Cost reduction':<30} {'':>12} {'':>12} "
      f"{(total_naive-total_bom)/total_naive*100:>10.1f}%")

# ── sensitivity table: €100–€500 in €50 steps ──────────────────────────────
print(f"\n{sep}")
print("  Sensitivity table — annual saving across unit cost range")
print(sep)
print(f"\n  {'Unit Cost':>10} {'SS Holding':>12} {'Emerg. PO':>11} "
      f"{'Stockout':>10} {'Total':>10} {'vs baseline':>12}")
base_saving = compute_savings(200)["total_saving"]
for uc_s in range(100, 550, 50):
    r = compute_savings(uc_s)
    vs_base = (r["total_saving"] - base_saving) / base_saving * 100
    marker = " ◄ baseline" if uc_s == 200 else ""
    print(f"  €{uc_s:>7,.0f}   €{r['c1_holding']:>9,.0f}   "
          f"€{r['c2_ordering']:>7,.0f}   "
          f"€{r['c3_stockout']:>7,.0f}   "
          f"€{r['total_saving']:>7,.0f}   "
          f"{vs_base:>+8.0f}%{marker}")

# ══════════════════════════════════════════════════════════════════════════════
#  SAVE DATA
# ══════════════════════════════════════════════════════════════════════════════
uc_range = np.arange(100, 510, 10)
rows = [compute_savings(uc) for uc in uc_range]
cost_df = pd.DataFrame(rows)
cost_df.to_csv(DATA_DIR / "cost_simulation.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
#  FIG 12 — Sensitivity chart: annual saving vs unit cost (€100–€500)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Step 6 — Cost Simulation: Annual Saving from BOM-Aware Forecasting",
             fontsize=13, fontweight="bold", y=1.02, color=C["text"])

# ── Left: total saving line + component stacking ──────────────────────────────
ax = axes[0]
ax.fill_between(cost_df["unit_cost"], 0, cost_df["c1_holding"],
                color=C["comp1"], alpha=0.65, label="SS Holding Cost Saving")
ax.fill_between(cost_df["unit_cost"],
                cost_df["c1_holding"],
                cost_df["c1_holding"] + cost_df["c2_ordering"],
                color=C["comp2"], alpha=0.65, label="Emergency Ordering Saving")
ax.fill_between(cost_df["unit_cost"],
                cost_df["c1_holding"] + cost_df["c2_ordering"],
                cost_df["total_saving"],
                color=C["comp3"], alpha=0.65, label="Stockout Exposure Saving")
ax.plot(cost_df["unit_cost"], cost_df["total_saving"],
        color=C["saving"], linewidth=2.5, zorder=5, label="Total Annual Saving")

# spot annotations at 3 price points
for uc_spot, offset_y in [(100, 120), (200, 120), (500, 120)]:
    r = compute_savings(uc_spot)
    ax.axvline(uc_spot, color=C["text"], linewidth=0.8,
               linestyle=":", alpha=0.4, zorder=1)
    ax.scatter([uc_spot], [r["total_saving"]], color=C["saving"],
               s=80, zorder=6)
    ax.annotate(f"€{r['total_saving']:,.0f}/yr",
                xy=(uc_spot, r["total_saving"]),
                xytext=(uc_spot + 8, r["total_saving"] + offset_y),
                fontsize=8.5, fontweight="bold", color=C["saving"],
                arrowprops=dict(arrowstyle="-", color=C["saving"],
                                lw=1.0, alpha=0.6))

ax.set_xlabel("Unit cost (€)")
ax.set_ylabel("Annual saving (€)")
ax.set_title("A  |  Annual saving by cost component\n(stacked area = component breakdown)",
             loc="left")
ax.legend(fontsize=8.5, loc="upper left")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
ax.set_xlim(100, 500)
ax.set_ylim(0, cost_df["total_saving"].max() * 1.2)

# ── Right: % saving vs baseline (€200) ────────────────────────────────────────
ax = axes[1]
base = compute_savings(200)["total_saving"]
pct_change = (cost_df["total_saving"] - base) / base * 100

ax.plot(cost_df["unit_cost"], cost_df["total_saving"],
        color=C["saving"], linewidth=2.5, label="Total Annual Saving")
ax.fill_between(cost_df["unit_cost"], cost_df["total_saving"],
                alpha=0.12, color=C["saving"])

# secondary y-axis: % vs baseline
ax2 = ax.twinx()
ax2.spines["top"].set_visible(False)
ax2.plot(cost_df["unit_cost"], pct_change,
         color=C["amber"], linewidth=1.5, linestyle="--",
         alpha=0.8, label="% vs €200 baseline")
ax2.set_ylabel("% change vs €200 baseline", color=C["amber"], fontsize=9)
ax2.tick_params(axis="y", colors=C["amber"], labelsize=8.5)
ax2.axhline(0, color=C["amber"], linewidth=0.8, linestyle=":", alpha=0.5)
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:+.0f}%"))

# label the 3 spotpoints
for uc_spot in [100, 200, 500]:
    r = compute_savings(uc_spot)
    ax.axvline(uc_spot, color=C["text"], linewidth=0.8,
               linestyle=":", alpha=0.35, zorder=1)

ax.set_xlabel("Unit cost (€)")
ax.set_ylabel("Annual saving (€)", color=C["saving"])
ax.tick_params(axis="y", colors=C["saving"])
ax.set_title("B  |  Sensitivity: total saving and % vs €200 baseline\n"
             "(primary axis = €; secondary = % relative to baseline)",
             loc="left")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
ax.set_xlim(100, 500)
ax.set_ylim(0, cost_df["total_saving"].max() * 1.2)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8.5, loc="upper left")

fig.text(0.5, -0.04,
         f"Assumptions: Z={PARAMS['Z']} (95% SL) | LT={PARAMS['lead_time_weeks']}wk "
         f"({PARAMS['lead_time_months']:.2f}mo) | Holding rate {PARAMS['holding_rate']*100:.0f}%/yr | "
         f"PO cost €{PARAMS['ordering_cost']} | {PARAMS['n_subs_per_year']} sub events/yr | "
         f"σ_demand proxied by MAE",
         ha="center", fontsize=7.8, color=C["text"], alpha=0.7, style="italic")

plt.tight_layout()
fig.savefig(PLOTS_DIR / "fig12_cost_sensitivity.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  → fig12_cost_sensitivity.png")

# ══════════════════════════════════════════════════════════════════════════════
#  FIG 13 — Cost breakdown: grouped bars at €100 / €200 / €500
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Annual Inventory Cost: Naive SES vs BOM-Aware SES",
             fontsize=13, fontweight="bold", y=1.02, color=C["text"])

spot_ucs   = [100, 200, 500]
spot_r     = [compute_savings(uc) for uc in spot_ucs]
x          = np.arange(len(spot_ucs))
w          = 0.30
labels_uc  = [f"€{uc}" for uc in spot_ucs]

# ── Left: stacked bars — naive vs BOM-aware absolute cost ─────────────────────
ax = axes[0]
naive_c1 = [Z * PARAMS["mae_naive"] * np.sqrt(LT) * uc * hr * n for uc in spot_ucs]
naive_c2 = [1 * oc * n for _ in spot_ucs]
naive_c3 = [PARAMS["mae_naive"] * ps * LT * uc * n for uc in spot_ucs]
bom_c1   = [Z * PARAMS["mae_bom"]   * np.sqrt(LT) * uc * hr * n for uc in spot_ucs]
bom_c2   = [(PARAMS["mae_bom"]/PARAMS["mae_naive"]) * oc * n for _ in spot_ucs]
bom_c3   = [PARAMS["mae_bom"]   * ps * LT * uc * n for uc in spot_ucs]

# Naive stacked
b1 = ax.bar(x - w/2, naive_c1, width=w, color=C["comp1"],
            alpha=0.85, label="SS Holding")
b2 = ax.bar(x - w/2, naive_c2, width=w, bottom=naive_c1,
            color=C["comp2"], alpha=0.85, label="Emergency PO")
b3 = ax.bar(x - w/2, naive_c3, width=w,
            bottom=[a+b for a,b in zip(naive_c1,naive_c2)],
            color=C["comp3"], alpha=0.85, label="Stockout")
# BOM stacked
ax.bar(x + w/2, bom_c1, width=w, color=C["comp1"], alpha=0.4, hatch="///")
ax.bar(x + w/2, bom_c2, width=w, bottom=bom_c1,
       color=C["comp2"], alpha=0.4, hatch="///")
ax.bar(x + w/2, bom_c3, width=w,
       bottom=[a+b for a,b in zip(bom_c1,bom_c2)],
       color=C["comp3"], alpha=0.4, hatch="///")

# total labels
for i, (uc, r) in enumerate(zip(spot_ucs, spot_r)):
    n_h = naive_c1[i]+naive_c2[i]+naive_c3[i]
    b_h = bom_c1[i]+bom_c2[i]+bom_c3[i]
    ax.text(i-w/2, n_h*1.02, f"€{n_h:,.0f}", ha="center",
            fontsize=8, fontweight="bold", color=C["naive"])
    ax.text(i+w/2, b_h*1.02, f"€{b_h:,.0f}", ha="center",
            fontsize=8, fontweight="bold", color=C["bom"])

ax.set_xticks(x)
ax.set_xticklabels(labels_uc)
ax.set_xlabel("Unit cost assumption")
ax.set_ylabel("Annual inventory cost (€)")
ax.set_title("A  |  Absolute annual cost — Naive (solid) vs BOM-aware (hatched)",
             loc="left")
legend_items = [
    mpatches.Patch(color=C["comp1"], alpha=0.85, label="SS Holding"),
    mpatches.Patch(color=C["comp2"], alpha=0.85, label="Emergency PO"),
    mpatches.Patch(color=C["comp3"], alpha=0.85, label="Stockout"),
    mpatches.Patch(facecolor="white", edgecolor=C["text"],
                   hatch="///", label="BOM-aware (hatched)"),
]
ax.legend(handles=legend_items, fontsize=8.5, loc="upper left")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"€{v:,.0f}"))

# ── Right: saving breakdown by component at 3 price points ────────────────────
ax = axes[1]
s_c1 = [r["c1_holding"]  for r in spot_r]
s_c2 = [r["c2_ordering"] for r in spot_r]
s_c3 = [r["c3_stockout"] for r in spot_r]
total_s = [r["total_saving"] for r in spot_r]

ax.bar(x, s_c1, width=0.5, color=C["comp1"], alpha=0.85, label="SS Holding Saving")
ax.bar(x, s_c2, width=0.5, bottom=s_c1,
       color=C["comp2"], alpha=0.85, label="Emerg. PO Saving")
ax.bar(x, s_c3, width=0.5, bottom=[a+b for a,b in zip(s_c1,s_c2)],
       color=C["comp3"], alpha=0.85, label="Stockout Saving")

for i, (tot, r) in enumerate(zip(total_s, spot_r)):
    ax.text(i, tot * 1.02, f"€{tot:,.0f}\n({tot/max(total_s)*100:.0f}%)",
            ha="center", fontsize=9, fontweight="bold", color=C["saving"])
    # component breakdown inside bars
    for j, (val, bot, label_str) in enumerate([
        (r["c1_holding"],  0,                           f"€{r['c1_holding']:,.0f}"),
        (r["c2_ordering"], r["c1_holding"],              f"€{r['c2_ordering']:,.0f}"),
        (r["c3_stockout"], r["c1_holding"]+r["c2_ordering"], f"€{r['c3_stockout']:,.0f}"),
    ]):
        if val > tot * 0.06:
            ax.text(i, bot + val/2, label_str,
                    ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels_uc)
ax.set_xlabel("Unit cost assumption")
ax.set_ylabel("Annual saving (€)")
ax.set_title("B  |  Annual saving by cost component at 3 price points",
             loc="left")
ax.legend(fontsize=8.5, loc="upper left")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"€{v:,.0f}"))
ax.set_ylim(0, max(total_s) * 1.22)

fig.text(0.5, -0.04,
         f"Parameter assumptions: Z=1.65 | LT=6wk | Holding rate=22%/yr | "
         f"PO cost=€100 | 26 substitution events/yr | σ_demand=MAE",
         ha="center", fontsize=7.8, color=C["text"], alpha=0.7, style="italic")

plt.tight_layout()
fig.savefig(PLOTS_DIR / "fig13_cost_breakdown.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → fig13_cost_breakdown.png")

# ── summary line ───────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print("  Summary")
print("="*72)
for r in spot_r:
    pct = r["total_saving"] / (Z * PARAMS["mae_naive"] * np.sqrt(LT) *
           r["unit_cost"] * hr * n +
           1 * oc * n +
           PARAMS["mae_naive"] * ps * LT * r["unit_cost"] * n) * 100
    print(f"  @ €{r['unit_cost']:<4}:  annual saving = €{r['total_saving']:>6,.0f}  "
          f"({pct:.1f}% of naive model's total transition-window cost)")

print(f"\n  Cost driver: SS holding cost is dominant above ~€150/unit.")
print(f"  Emergency PO saving (€{spot_r[0]['c2_ordering']:.0f}/yr) is unit-price independent —")
print(f"  it's the floor saving regardless of component price.")
print(f"\n  Files written:")
for f in ["cost_simulation.csv"]:
    size = (DATA_DIR / f).stat().st_size / 1024
    print(f"    data/{f}  ({size:.1f} KB)")
print("\nCost simulation complete. ✓")
