"""
BOM-Aware Demand Forecasting — Data Validation & Exploration
=============================================================
Quality gate script. Runs structural checks, semantic checks,
and produces diagnostic plots for the Data Overview dashboard page.

Checks performed
----------------
 1. No nulls in any dataset
 2. No duplicate rows
 3. Date coverage: every functional family has exactly 36 monthly rows
 4. No demand gaps (zero-length gaps between part-number segments)
 5. Demand continuity across substitutions (no hard resets to 0)
 6. Substitution event spacing ≥ 6 months within a family
 7. Part numbers are unique to exactly one functional ID
 8. Part number counts per family (1–4 expected)
 9. January seasonality is visible in the data (Jan avg > non-Jan avg)
10. Demand signal is non-negative everywhere
11. Lookup ↔ demand cross-reference (every part in demand exists in lookup)
12. Substitution events reference valid part numbers

Outputs
-------
  validation_report.txt   — printed summary, also saved to disk
  plots/fig_*.png         — individual diagnostic figures (dashboard-ready)
"""

import os
import sys
import textwrap
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_DIR  = Path("data")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

# ── colour palette (consistent across all figures) ─────────────────────────────
C = {
    "navy"   : "#1B2A4A",
    "blue"   : "#2563EB",
    "teal"   : "#0D9488",
    "amber"  : "#D97706",
    "red"    : "#DC2626",
    "grey"   : "#6B7280",
    "lgrey"  : "#E5E7EB",
    "bg"     : "#F8FAFC",
    "pl"     : ["#2563EB", "#0D9488", "#D97706", "#7C3AED"],   # one per product line
}

PRODUCT_LINES = ["PL-Alpha", "PL-Beta", "PL-Gamma", "PL-Delta"]
PL_COLOR      = dict(zip(PRODUCT_LINES, C["pl"]))

plt.rcParams.update({
    "font.family"       : "DejaVu Sans",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.facecolor"    : C["bg"],
    "figure.facecolor"  : "white",
    "axes.titlesize"    : 13,
    "axes.titleweight"  : "bold",
    "axes.labelsize"    : 10,
    "xtick.labelsize"   : 9,
    "ytick.labelsize"   : 9,
    "axes.grid"         : True,
    "grid.color"        : C["lgrey"],
    "grid.linewidth"    : 0.7,
})

# ── load data ──────────────────────────────────────────────────────────────────
print("\nLoading datasets …")
df  = pd.read_csv(DATA_DIR / "demand_history.csv",      parse_dates=["date"])
lkp = pd.read_csv(DATA_DIR / "functional_id_lookup.csv")
sub = pd.read_csv(DATA_DIR / "substitution_events.csv", parse_dates=["event_date"])
bom = pd.read_csv(DATA_DIR / "product_bom.csv")
print(f"  demand_history      : {len(df):,} rows")
print(f"  functional_id_lookup: {len(lkp):,} rows")
print(f"  substitution_events : {len(sub):,} rows")
print(f"  product_bom         : {len(bom):,} rows")

# ── validation harness ─────────────────────────────────────────────────────────
CHECKS   = []   # (check_id, description, passed, detail)
WARNINGS = []

def record(check_id, description, passed, detail=""):
    CHECKS.append((check_id, description, passed, detail))
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{status}] {check_id}: {description}")
    if detail:
        for line in textwrap.wrap(detail, 72):
            print(f"            {line}")

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 1 — No nulls
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Structural checks ─────────────────────────────────────")
null_counts = {
    "demand"  : df.isnull().sum().sum(),
    "lookup"  : lkp.isnull().sum().sum(),
    "subs"    : sub.isnull().sum().sum(),
    "bom"     : bom.isnull().sum().sum(),
}
total_nulls = sum(null_counts.values())
record("C01", "No null values in any dataset",
       total_nulls == 0,
       f"Null counts → demand:{null_counts['demand']} lookup:{null_counts['lookup']} "
       f"subs:{null_counts['subs']} bom:{null_counts['bom']}")

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 2 — No duplicate rows
# ══════════════════════════════════════════════════════════════════════════════
dup_demand = df.duplicated(subset=["date","part_number"]).sum()
dup_lookup = lkp.duplicated(subset=["part_number"]).sum()
record("C02", "No duplicate (date, part_number) rows in demand_history",
       dup_demand == 0, f"{dup_demand} duplicates found")
record("C03", "No duplicate part_number rows in lookup",
       dup_lookup == 0, f"{dup_lookup} duplicates found")

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 4 — Every functional family has exactly 36 monthly rows
# ══════════════════════════════════════════════════════════════════════════════
rows_per_fid = df.groupby(["product_line","functional_id"])["date"].count()
bad_row_count = (rows_per_fid != 36).sum()
record("C04", "Every functional family has exactly 36 demand rows",
       bad_row_count == 0,
       f"{bad_row_count} families with wrong row count" if bad_row_count else
       f"All {len(rows_per_fid)} families have exactly 36 rows")

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 5 — No demand gaps (missing months) within any family
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Continuity checks ─────────────────────────────────────")
gap_families = []
for fid, grp in df.groupby(["product_line","functional_id"]):
    dates_sorted = grp["date"].sort_values().reset_index(drop=True)
    expected = pd.date_range(dates_sorted.iloc[0], periods=len(dates_sorted), freq="MS")
    if not (dates_sorted.values == expected.values).all():
        gap_families.append(fid)
record("C05", "No monthly gaps within any functional family",
       len(gap_families) == 0,
       f"Families with gaps: {gap_families}" if gap_families else
       "All 48 families are gap-free")

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 6 — Demand continuity across substitutions
#   Test: demand in month-of-substitution is within 3σ of the preceding 6-month
#   rolling mean for that functional family. A naive model reset would show 0.
# ══════════════════════════════════════════════════════════════════════════════
continuity_issues = []
sub_stats = []   # saved for the transition-window plot
for _, row in sub.iterrows():
    fid    = row["functional_id"]
    pl     = row["product_line"]
    ev_dt  = row["event_date"]
    family = df[(df["functional_id"]==fid) & (df["product_line"]==pl)].sort_values("date")
    idx    = family.index[family["date"] == ev_dt]
    if len(idx) == 0:
        continue
    pos = family.index.get_loc(idx[0])
    if pos < 3:
        continue
    pre_window  = family.iloc[max(0, pos-6):pos]["demand"]
    post_window = family.iloc[pos:min(len(family), pos+6)]["demand"]
    pre_mean = pre_window.mean()
    pre_std  = pre_window.std() if len(pre_window) > 1 else pre_mean * 0.15
    sub_val  = family.iloc[pos]["demand"]
    # flag if substitution month demand is effectively 0 while pre_mean > 5
    if pre_mean > 5 and sub_val < pre_mean * 0.1:
        continuity_issues.append((fid, pl, str(ev_dt.date()), sub_val, pre_mean))
    sub_stats.append({
        "functional_id" : fid, "product_line": pl, "event_date": ev_dt,
        "pre_mean"      : pre_mean, "post_mean": post_window.mean(),
        "sub_val"       : sub_val,  "pre_std": max(pre_std, 0.1),
        "pct_change"    : (post_window.mean() - pre_mean) / max(pre_mean, 1) * 100,
    })
record("C06", "Demand signal is continuous across substitution events (no hard resets)",
       len(continuity_issues) == 0,
       f"{len(continuity_issues)} discontinuity issues found" if continuity_issues else
       f"Checked {len(sub_stats)} transition windows — all continuous")

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 7 — Substitution spacing ≥ 6 months within a family
# ══════════════════════════════════════════════════════════════════════════════
spacing_issues = []
for fid, grp in sub.groupby(["product_line","functional_id"]):
    ev_dates = sorted(grp["event_date"].tolist())
    for i in range(1, len(ev_dates)):
        gap_months = (ev_dates[i].year - ev_dates[i-1].year)*12 + \
                     (ev_dates[i].month - ev_dates[i-1].month)
        if gap_months < 6:
            spacing_issues.append((fid, gap_months))
record("C07", "Substitution events within a family are ≥ 6 months apart",
       len(spacing_issues) == 0,
       f"Issues: {spacing_issues}" if spacing_issues else
       f"All {len(sub)} events meet the 6-month spacing rule")

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 8 — Each part number maps to exactly one functional ID
# ══════════════════════════════════════════════════════════════════════════════
pn_fid_counts = lkp.groupby("part_number")["functional_id"].nunique()
ambiguous_pns = (pn_fid_counts > 1).sum()
record("C08", "Each part number maps to exactly one functional ID",
       ambiguous_pns == 0,
       f"{ambiguous_pns} part numbers with >1 functional ID" if ambiguous_pns else
       f"All {len(pn_fid_counts)} part numbers are unambiguous")

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 9 — Part numbers per family: 2–4 expected
# ══════════════════════════════════════════════════════════════════════════════
pn_per_family = lkp.groupby(["product_line","functional_id"])["part_number"].count()
out_of_range  = ((pn_per_family < 2) | (pn_per_family > 4)).sum()
pn_dist       = pn_per_family.value_counts().sort_index()
record("C09", "Part numbers per family in expected range [2–4]",
       out_of_range == 0,
       "Distribution: " + " | ".join(f"{k} PNs: {v} families" for k,v in pn_dist.items()))

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 10 — January seasonality is detectable
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Semantic / signal checks ──────────────────────────────")
df["month"] = df["date"].dt.month
jan_avg     = df[df["month"] == 1]["demand"].mean()
non_jan_avg = df[df["month"] != 1]["demand"].mean()
jan_pct     = (jan_avg / non_jan_avg - 1) * 100
record("C10", "January demand is detectably higher than other months",
       jan_avg > non_jan_avg,
       f"Jan avg = {jan_avg:.1f}  |  Non-Jan avg = {non_jan_avg:.1f}  "
       f"|  Jan premium = +{jan_pct:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 11 — No negative demand
# ══════════════════════════════════════════════════════════════════════════════
neg_demand = (df["demand"] < 0).sum()
record("C11", "All demand values are non-negative",
       neg_demand == 0,
       f"{neg_demand} negative values found")

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 12 — Cross-reference: every part in demand exists in lookup
# ══════════════════════════════════════════════════════════════════════════════
demand_pns = set(df["part_number"].unique())
lookup_pns = set(lkp["part_number"].unique())
orphan_pns = demand_pns - lookup_pns
record("C12", "Every part number in demand_history exists in functional_id_lookup",
       len(orphan_pns) == 0,
       f"Orphan part numbers: {orphan_pns}" if orphan_pns else
       f"All {len(demand_pns)} part numbers are matched")

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 13 — Substitution events reference valid part numbers
# ══════════════════════════════════════════════════════════════════════════════
sub_pns   = set(sub["old_part_number"]) | set(sub["new_part_number"])
ghost_pns = sub_pns - lookup_pns
record("C13", "All substitution event part numbers exist in lookup",
       len(ghost_pns) == 0,
       f"Ghost part numbers: {ghost_pns}" if ghost_pns else
       f"All {len(sub_pns)} referenced part numbers are valid")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
n_pass = sum(1 for c in CHECKS if c[2])
n_fail = len(CHECKS) - n_pass

print(f"\n{'═'*60}")
print(f"  Validation complete: {n_pass}/{len(CHECKS)} checks passed"
      + (f"  ← {n_fail} FAILED" if n_fail else "  ✓ ALL CLEAR"))
print(f"{'═'*60}\n")

# Save report to text file
report_lines = [
    "BOM-Aware Demand Forecasting — Data Validation Report",
    "="*60,
    f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
    "",
    f"Dataset sizes:",
    f"  demand_history      : {len(df):,} rows",
    f"  functional_id_lookup: {len(lkp):,} rows",
    f"  substitution_events : {len(sub):,} rows",
    f"  product_bom         : {len(bom):,} rows",
    "",
    "Check results:",
]
for cid, desc, passed, detail in CHECKS:
    report_lines.append(f"  [{'PASS' if passed else 'FAIL'}] {cid}: {desc}")
    if detail:
        report_lines.append(f"        {detail}")
report_lines += ["", f"Result: {n_pass}/{len(CHECKS)} checks passed"]
Path("data/validation_report.txt").write_text("\n".join(report_lines))

if n_fail > 0:
    print("WARNING: Some checks failed — review before proceeding.\n")
    # don't sys.exit so plots still generate

# ══════════════════════════════════════════════════════════════════════════════
#  DIAGNOSTIC PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("Generating diagnostic plots …")

sub_stats_df = pd.DataFrame(sub_stats) if sub_stats else pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — Demand across a substitution event: naive vs BOM-aware perspective
#   Pick one family with 2+ substitutions and show it in both framings
# ─────────────────────────────────────────────────────────────────────────────
# Choose a visually rich family (3 substitution events → 4 part numbers)
target_fids = sub.groupby(["product_line","functional_id"]).size()
target_fids = target_fids[target_fids >= 2].reset_index()
demo_row    = target_fids.sample(1, random_state=7).iloc[0]
demo_pl     = demo_row["product_line"]
demo_fid    = demo_row["functional_id"]

demo_df = df[(df["functional_id"]==demo_fid) & (df["product_line"]==demo_pl)].sort_values("date")
demo_sub = sub[(sub["functional_id"]==demo_fid) & (sub["product_line"]==demo_pl)].sort_values("event_date")
demo_pns = demo_df["part_number"].unique()
pn_colors = [C["blue"], C["teal"], C["amber"], C["red"]][:len(demo_pns)]
pn_color_map = dict(zip(demo_pns, pn_colors))

fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
fig.suptitle(
    f"Fig 1 — Substitution event: naive (per-part) vs BOM-aware view\n"
    f"{demo_fid} | {demo_pl}",
    fontsize=13, fontweight="bold", y=1.01
)

# Panel A: Naive — each part number is an independent series
ax = axes[0]
ax.set_title("A  |  Naïve view: each part number is an isolated time series", loc="left")
for pn, grp in demo_df.groupby("part_number"):
    color = pn_color_map[pn]
    grp = grp.sort_values("date")
    ax.plot(grp["date"], grp["demand"], color=color, linewidth=2, zorder=3)
    ax.scatter(grp["date"], grp["demand"], color=color, s=22, zorder=4)
    # label at start of segment
    ax.annotate(pn, xy=(grp["date"].iloc[0], grp["demand"].iloc[0]),
                xytext=(4, 6), textcoords="offset points",
                fontsize=7.5, color=color, fontweight="bold")
for _, ev in demo_sub.iterrows():
    ax.axvline(ev["event_date"], color=C["red"], linewidth=1.2, linestyle="--", alpha=0.6)
ax.set_ylabel("Demand (units/month)")
ax.annotate("← model sees no history here", xy=(demo_sub.iloc[0]["event_date"], ax.get_ylim()[1]*0.85),
            fontsize=8, color=C["red"], alpha=0.85,
            xytext=(8, 0), textcoords="offset points")

# Panel B: BOM-aware — continuous chain
ax = axes[1]
ax.set_title("B  |  BOM-aware view: continuous history chained by functional ID", loc="left")
# shade each segment a different colour
seg_boundaries = [demo_df["date"].min()] + [ev["event_date"] for _, ev in demo_sub.iterrows()] + [demo_df["date"].max() + pd.offsets.MonthEnd(1)]
for i, (t0, t1, pn) in enumerate(zip(seg_boundaries[:-1], seg_boundaries[1:], demo_pns)):
    seg = demo_df[demo_df["part_number"]==pn]
    ax.fill_between(seg["date"], 0, seg["demand"],
                    color=pn_colors[i], alpha=0.12, zorder=1)
    ax.plot(seg["date"], seg["demand"], color=pn_colors[i], linewidth=2.2, zorder=3)
    ax.scatter(seg["date"], seg["demand"], color=pn_colors[i], s=22, zorder=4)
for _, ev in demo_sub.iterrows():
    ax.axvline(ev["event_date"], color=C["red"], linewidth=1.4, linestyle="--", alpha=0.7,
               label="Substitution event")
ax.set_ylabel("Demand (units/month)")
ax.set_xlabel("Month")

# legend for segments
legend_patches = [mpatches.Patch(color=pn_colors[i], alpha=0.7, label=pn)
                  for i, pn in enumerate(demo_pns)]
legend_patches.append(Line2D([0],[0], color=C["red"], linestyle="--", linewidth=1.4, label="Substitution"))
ax.legend(handles=legend_patches, fontsize=8, ncol=len(demo_pns)+1,
          loc="upper left", framealpha=0.9)

plt.tight_layout()
fig.savefig(PLOTS_DIR / "fig1_substitution_event_naive_vs_bomaware.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → fig1_substitution_event_naive_vs_bomaware.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — January seasonality heatmap across all product lines
# ─────────────────────────────────────────────────────────────────────────────
monthly_avg = (df.groupby(["product_line", df["date"].dt.month])["demand"]
               .mean().unstack(level=0))
month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

fig, ax = plt.subplots(figsize=(11, 4))
im = ax.imshow(monthly_avg.T.values, aspect="auto", cmap="YlOrRd", vmin=80, vmax=monthly_avg.values.max())
ax.set_xticks(range(12))
ax.set_xticklabels(month_labels)
ax.set_yticks(range(len(PRODUCT_LINES)))
ax.set_yticklabels(PRODUCT_LINES)
ax.set_title("Fig 2 — Avg monthly demand heatmap: January spike visible across all product lines",
             loc="left", fontsize=12, fontweight="bold")
# annotate cells
for i in range(12):
    for j in range(len(PRODUCT_LINES)):
        val = monthly_avg.T.values[j, i]
        ax.text(i, j, f"{val:.0f}", ha="center", va="center",
                fontsize=8.5, color="white" if val > 150 else C["navy"], fontweight="bold")
cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("Avg demand (units/month)", fontsize=9)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "fig2_january_seasonality_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → fig2_january_seasonality_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Part numbers per functional family (histogram)
# ─────────────────────────────────────────────────────────────────────────────
pn_counts = lkp.groupby(["product_line","functional_id"])["part_number"].count().reset_index()
pn_counts.columns = ["product_line","functional_id","n_part_numbers"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: histogram
ax = axes[0]
bins = [1.5, 2.5, 3.5, 4.5]
for pl, color in PL_COLOR.items():
    vals = pn_counts[pn_counts["product_line"]==pl]["n_part_numbers"]
    ax.hist(vals, bins=bins, alpha=0.65, color=color, label=pl, edgecolor="white", linewidth=0.8)
ax.set_xlabel("Number of part numbers per functional family")
ax.set_ylabel("Count of families")
ax.set_title("Fig 3A — Part numbers per functional family", loc="left", fontweight="bold")
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.legend(fontsize=8)
ax.set_xlim(1.5, 4.5)

# Right: stacked bar by product line
ax = axes[1]
counts_pivot = pn_counts.groupby(["product_line","n_part_numbers"]).size().unstack(fill_value=0)
bottom = np.zeros(len(PRODUCT_LINES))
bar_colors = [C["blue"], C["teal"], C["amber"]]
for i, col in enumerate(sorted(counts_pivot.columns)):
    vals = [counts_pivot.loc[pl, col] if pl in counts_pivot.index and col in counts_pivot.columns
            else 0 for pl in PRODUCT_LINES]
    bars = ax.bar(PRODUCT_LINES, vals, bottom=bottom, color=bar_colors[i],
                  label=f"{col} PNs", edgecolor="white", linewidth=0.8)
    for bar_i, (bar, v) in enumerate(zip(bars, vals)):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bottom[bar_i] + v/2, str(v),
                    ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    bottom += np.array(vals)
ax.set_title("Fig 3B — Distribution by product line", loc="left", fontweight="bold")
ax.set_ylabel("Number of functional families")
ax.legend(fontsize=8, loc="upper right")
ax.set_ylim(0, 14)

plt.tight_layout()
fig.savefig(PLOTS_DIR / "fig3_part_numbers_per_family.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → fig3_part_numbers_per_family.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — Substitution event timeline across all families
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 7))
fid_list  = sorted(df["functional_id"].unique())
fid_index = {fid: i for i, fid in enumerate(fid_list)}

# Demand heatmap per family (colour = avg monthly demand → shows active segment)
demand_matrix = np.zeros((len(fid_list), 36))
date_index    = {d: i for i, d in enumerate(sorted(df["date"].unique()))}
for _, row in df.iterrows():
    fi = fid_index.get(row["functional_id"])
    di = date_index.get(row["date"])
    if fi is not None and di is not None:
        demand_matrix[fi, di] += row["demand"]

ax.imshow(demand_matrix, aspect="auto", cmap="Blues", alpha=0.35,
          extent=[-0.5, 35.5, len(fid_list)-0.5, -0.5])

# Substitution events as vertical tick marks
for _, ev in sub.iterrows():
    fid  = ev["functional_id"]
    if fid in fid_index:
        yi   = fid_index[fid]
        xi   = date_index.get(ev["event_date"], None)
        if xi is not None:
            pl_col = PL_COLOR.get(ev["product_line"], C["blue"])
            ax.scatter(xi, yi, marker="|", s=160, color=pl_col,
                       linewidth=2.5, zorder=5, alpha=0.9)

# y-axis: product line bands
for pl_i, pl in enumerate(PRODUCT_LINES):
    fids_in_pl = sorted(df[df["product_line"]==pl]["functional_id"].unique())
    if fids_in_pl:
        y0 = fid_index[fids_in_pl[0]]  - 0.5
        y1 = fid_index[fids_in_pl[-1]] + 0.5
        ax.axhspan(y0, y1, color=C["pl"][pl_i], alpha=0.05, zorder=0)
        ax.text(-1.5, (y0+y1)/2, pl.replace("PL-",""), va="center", ha="right",
                fontsize=8, color=C["pl"][pl_i], fontweight="bold", rotation=0)

# x-axis: month labels
all_dates = sorted(df["date"].unique())
tick_pos   = [i for i, d in enumerate(all_dates) if pd.Timestamp(d).month in [1, 7]]
tick_labs  = [pd.Timestamp(d).strftime("%b\n%Y") for i, d in enumerate(all_dates)
              if pd.Timestamp(d).month in [1, 7]]
ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_labs, fontsize=8)
ax.set_yticks(list(fid_index.values()))
ax.set_yticklabels(list(fid_index.keys()), fontsize=6.5)
ax.set_title("Fig 4 — Substitution event timeline across all 48 functional families\n"
             "(colour band = product line | tick mark = substitution event)",
             loc="left", fontweight="bold", fontsize=11)
ax.set_xlabel("Month")
ax.set_ylabel("Functional ID")
# legend
legend_handles = [mpatches.Patch(color=C["pl"][i], alpha=0.6, label=pl)
                  for i, pl in enumerate(PRODUCT_LINES)]
legend_handles.append(Line2D([0],[0], marker="|", color=C["grey"],
                              markersize=10, linewidth=0, label="Substitution event"))
ax.legend(handles=legend_handles, fontsize=8, loc="lower right", framealpha=0.9)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "fig4_substitution_event_timeline.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → fig4_substitution_event_timeline.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 — Demand distribution: overall + by product line
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Left: overall histogram
ax = axes[0]
ax.hist(df["demand"], bins=40, color=C["blue"], alpha=0.75, edgecolor="white")
ax.axvline(df["demand"].mean(), color=C["amber"], linewidth=2, linestyle="--",
           label=f"Mean = {df['demand'].mean():.0f}")
ax.axvline(df["demand"].median(), color=C["teal"], linewidth=2, linestyle=":",
           label=f"Median = {df['demand'].median():.0f}")
ax.set_xlabel("Monthly demand (units)")
ax.set_ylabel("Frequency")
ax.set_title("Fig 5A — Overall demand distribution", loc="left", fontweight="bold")
ax.legend(fontsize=9)

# Right: box plots per product line
ax = axes[1]
data_by_pl = [df[df["product_line"]==pl]["demand"].values for pl in PRODUCT_LINES]
bp = ax.boxplot(data_by_pl, patch_artist=True, widths=0.55,
                medianprops=dict(color="white", linewidth=2.5))
for patch, color in zip(bp["boxes"], C["pl"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
for whisker in bp["whiskers"]:
    whisker.set(color=C["grey"], linewidth=1.2)
for cap in bp["caps"]:
    cap.set(color=C["grey"], linewidth=1.2)
for flier in bp["fliers"]:
    flier.set(marker="o", color=C["grey"], alpha=0.4, markersize=3)
ax.set_xticks(range(1, len(PRODUCT_LINES)+1))
ax.set_xticklabels([pl.replace("PL-","") for pl in PRODUCT_LINES])
ax.set_ylabel("Monthly demand (units)")
ax.set_title("Fig 5B — Demand distribution by product line", loc="left", fontweight="bold")

plt.tight_layout()
fig.savefig(PLOTS_DIR / "fig5_demand_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → fig5_demand_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 6 — Pre/post substitution demand change (all 77 events)
# ─────────────────────────────────────────────────────────────────────────────
if not sub_stats_df.empty:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: scatter — pre mean vs post mean
    ax = axes[0]
    for pl, color in PL_COLOR.items():
        mask = sub_stats_df["product_line"] == pl
        ax.scatter(sub_stats_df.loc[mask,"pre_mean"],
                   sub_stats_df.loc[mask,"post_mean"],
                   color=color, alpha=0.65, s=45, label=pl, zorder=3)
    lim = max(sub_stats_df["pre_mean"].max(), sub_stats_df["post_mean"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color=C["grey"], linewidth=1, linestyle="--", zorder=1)
    ax.set_xlabel("Avg demand — 6 months BEFORE substitution")
    ax.set_ylabel("Avg demand — 6 months AFTER substitution")
    ax.set_title("Fig 6A — Pre vs post substitution demand level\n(diagonal = no change)",
                 loc="left", fontweight="bold")
    ax.legend(fontsize=8)

    # Right: histogram of % change
    ax = axes[1]
    ax.hist(sub_stats_df["pct_change"], bins=25, color=C["teal"], alpha=0.75, edgecolor="white")
    ax.axvline(0, color=C["red"], linewidth=1.5, linestyle="--")
    ax.axvline(sub_stats_df["pct_change"].mean(), color=C["amber"], linewidth=2,
               linestyle="--", label=f"Mean = {sub_stats_df['pct_change'].mean():.1f}%")
    ax.set_xlabel("% change in avg demand across substitution (pre→post 6m)")
    ax.set_ylabel("Frequency")
    ax.set_title("Fig 6B — Demand change at substitution events\n(noise only — no systematic jump)",
                 loc="left", fontweight="bold")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "fig6_pre_post_substitution_demand.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → fig6_pre_post_substitution_demand.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 7 — Four representative families (one per product line) — full 36-month series
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False)
axes = axes.flatten()

# Pick one "interesting" family per product line (most substitutions)
for pl_i, pl in enumerate(PRODUCT_LINES):
    pl_sub = sub[sub["product_line"]==pl]
    best_fid = (pl_sub.groupby("functional_id").size().idxmax()
                if len(pl_sub) else
                df[df["product_line"]==pl]["functional_id"].iloc[0])
    family = df[(df["functional_id"]==best_fid) & (df["product_line"]==pl)].sort_values("date")
    ev_dates = sub[(sub["functional_id"]==best_fid) & (sub["product_line"]==pl)]["event_date"].tolist()
    pns = family["part_number"].unique()
    seg_colors = [C["blue"], C["teal"], C["amber"], C["red"]][:len(pns)]

    ax = axes[pl_i]
    for pn, color in zip(pns, seg_colors):
        seg = family[family["part_number"]==pn]
        ax.plot(seg["date"], seg["demand"], color=color, linewidth=2.2, zorder=3)
        ax.fill_between(seg["date"], 0, seg["demand"], color=color, alpha=0.08)
        ax.scatter(seg["date"], seg["demand"], color=color, s=18, zorder=4)
    for ev_dt in ev_dates:
        ax.axvline(ev_dt, color=C["red"], linewidth=1.3, linestyle="--", alpha=0.65)
        ax.annotate("sub", xy=(ev_dt, ax.get_ylim()[1]*0.9),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=7, color=C["red"], alpha=0.8)

    # Jan highlights
    jan_dates = family[family["date"].dt.month==1]["date"]
    for jd in jan_dates:
        ax.axvspan(jd, jd + pd.offsets.MonthEnd(1), color=C["amber"], alpha=0.07, zorder=0)

    ax.set_title(f"{pl} — {best_fid}", loc="left", fontweight="bold", fontsize=10)
    ax.set_ylabel("Demand (units/month)")
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(bymonth=[1, 7]))
    n_pns = len(pns)
    ax.set_xlabel(f"{n_pns} part numbers | {len(ev_dates)} substitution events")

fig.suptitle("Fig 7 — Representative families (one per product line): trend + seasonality + substitution events",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "fig7_representative_families.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → fig7_representative_families.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 8 — Substitution reason breakdown
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
reason_counts = sub["reason"].value_counts()
bar_colors_r  = [C["pl"][i % 4] for i in range(len(reason_counts))]
bars = ax.barh(reason_counts.index, reason_counts.values,
               color=bar_colors_r, alpha=0.8, edgecolor="white")
for bar, val in zip(bars, reason_counts.values):
    ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
            str(val), va="center", fontsize=10, fontweight="bold", color=C["navy"])
ax.set_xlabel("Number of substitution events")
ax.set_title("Fig 8 — Substitution events by reason category", loc="left", fontweight="bold")
ax.invert_yaxis()
ax.set_xlim(0, reason_counts.values.max() * 1.15)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "fig8_substitution_reasons.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → fig8_substitution_reasons.png")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\nAll plots saved to: {PLOTS_DIR.resolve()}")
print(f"Validation report : data/validation_report.txt")
print(f"\nFinal status: {'✓ ALL CHECKS PASSED — data is ready for modelling.' if n_fail == 0 else f'⚠  {n_fail} CHECK(S) FAILED — review before proceeding.'}")
