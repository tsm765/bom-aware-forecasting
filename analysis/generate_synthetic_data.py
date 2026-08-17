"""
BOM-Aware Demand Forecasting Under Component Churn
===================================================
Synthetic Data Generation Script
----------------------------------
Generates:
  - demand_history.csv     : monthly demand per part number
  - functional_id_lookup.csv: part number → functional ID mapping
  - substitution_events.csv : log of substitution events (who replaced whom, when)
  - product_bom.csv        : product line → component (functional ID) mapping

Design decisions
----------------
* 4 product lines, 12 components each → 48 functional component families
* Each component family has 1–3 substitution events over 36 months
  (so a family may have 2–4 distinct part numbers in sequence)
* Demand signal = trend + January seasonality spike + white noise
  - Trend: weak positive (+0.5 units/month) or flat, varies per family
  - Seasonality: January gets a +20–40% spike (maintenance planning season)
  - Noise: Gaussian, std ≈ 15% of base demand
* On substitution, the NEW part number inherits the exact same underlying
  demand signal (same functional need), but its own part-number-level time
  series starts from zero — this is the core problem we demonstrate
* Base demand levels: 10–200 units/month, drawn once per family and held
  (with trend) across its full lifetime

Column definitions
------------------
demand_history.csv:
  date            : first day of month (YYYY-MM-DD)
  part_number     : SKU / component part number
  functional_id   : shared functional role identifier
  product_line    : which product line consumes this component
  demand          : monthly demand (units, integers ≥ 0)
  is_active       : 1 if this part number was the active BOM part that month

functional_id_lookup.csv:
  part_number     : SKU
  functional_id   : shared functional role
  product_line    : product line
  active_from     : month this part became active (YYYY-MM-DD)
  active_to       : month this part was deactivated (YYYY-MM-DD or "current")

substitution_events.csv:
  event_date      : first month the new part took over (YYYY-MM-DD)
  functional_id   : the functional role
  product_line    : product line
  old_part_number : part being replaced
  new_part_number : replacement part
  reason          : short reason string (e.g. "EOL", "cost reduction", "design change")
"""

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import random
import string
import os

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
N_PRODUCT_LINES     = 4
N_COMPONENTS_EACH   = 12          # functional component families per product line
N_MONTHS            = 36
START_DATE          = pd.Timestamp("2022-01-01")
SUBSTITUTION_RANGE  = (1, 3)      # min/max substitution events per family

PRODUCT_LINES = ["PL-Alpha", "PL-Beta", "PL-Gamma", "PL-Delta"]

SUBSTITUTION_REASONS = [
    "EOL (supplier discontinued)",
    "Cost reduction",
    "Design change",
    "Supply shortage / emergency sub",
    "Regulatory compliance",
]

# Demand parameters (units/month)
BASE_DEMAND_LOW  = 10
BASE_DEMAND_HIGH = 200
TREND_CHOICES    = [0.0, 0.0, 0.5, 1.0]   # most flat, some growing
JAN_SPIKE_RANGE  = (0.20, 0.40)            # +20 to +40% in January
NOISE_FRACTION   = 0.15                    # Gaussian noise std as fraction of base

# ─────────────────────────────────────────────
# Helper: generate a random part number
# ─────────────────────────────────────────────
_used_part_numbers: set = set()

def new_part_number(prefix: str) -> str:
    """Generate a unique part number like PL-A-1042-AB"""
    while True:
        digits = rng.integers(1000, 9999)
        suffix = "".join(random.choices(string.ascii_uppercase, k=2))
        pn = f"{prefix}-{digits}-{suffix}"
        if pn not in _used_part_numbers:
            _used_part_numbers.add(pn)
            return pn

# ─────────────────────────────────────────────
# Helper: demand signal for one family
# ─────────────────────────────────────────────
def demand_signal(base: float, trend: float, n_months: int, dates: list) -> np.ndarray:
    """
    Returns array of floats representing underlying monthly demand.
    Not yet rounded — callers round to int and clip at 0.
    """
    months = np.arange(n_months)
    trend_component = trend * months
    seasonal = np.array([
        base * rng.uniform(*JAN_SPIKE_RANGE) if d.month == 1 else 0.0
        for d in dates
    ])
    noise = rng.normal(0, base * NOISE_FRACTION, n_months)
    raw = base + trend_component + seasonal + noise
    return np.clip(raw, 0, None)

# ─────────────────────────────────────────────
# Build date index
# ─────────────────────────────────────────────
dates = [START_DATE + relativedelta(months=i) for i in range(N_MONTHS)]

# ─────────────────────────────────────────────
# Main generation loop
# ─────────────────────────────────────────────
demand_rows       = []
lookup_rows       = []
substitution_rows = []
bom_rows          = []

for pl_idx, pl in enumerate(PRODUCT_LINES):
    pl_prefix = pl.split("-")[1][:2].upper()   # "AL", "BE", "GA", "DE"

    for comp_idx in range(N_COMPONENTS_EACH):
        fid = f"FID-{pl_prefix}-{comp_idx+1:02d}"

        # --- demand signal parameters for this family ---
        base   = float(rng.integers(BASE_DEMAND_LOW, BASE_DEMAND_HIGH + 1))
        trend  = float(rng.choice(TREND_CHOICES))
        signal = demand_signal(base, trend, N_MONTHS, dates)

        # --- decide substitution schedule ---
        n_subs = int(rng.integers(SUBSTITUTION_RANGE[0], SUBSTITUTION_RANGE[1] + 1))

        # Pick substitution months; keep them spread out (at least 6 months apart,
        # not in first 3 or last 3 months so we always have pre/post windows)
        candidate_months = list(range(4, N_MONTHS - 4))
        sub_months_idx = sorted(
            rng.choice(candidate_months, size=min(n_subs, len(candidate_months)), replace=False).tolist()
        )
        # Enforce ≥6 month gap
        filtered = []
        last = -99
        for m in sub_months_idx:
            if m - last >= 6:
                filtered.append(m)
                last = m
        sub_months_idx = filtered[:3]    # cap at 3 actual events

        # Build segments: list of (start_month_idx, end_month_idx_exclusive, part_number)
        segments = []
        part_seq_num = 1
        seg_start = 0
        for sm in sub_months_idx:
            pn = new_part_number(f"{pl_prefix}{comp_idx+1:02d}")
            segments.append((seg_start, sm, pn))
            seg_start = sm
            part_seq_num += 1
        pn = new_part_number(f"{pl_prefix}{comp_idx+1:02d}")   # last (current) part
        segments.append((seg_start, N_MONTHS, pn))

        # --- populate rows ---
        # BOM row (one per functional component)
        bom_rows.append({
            "product_line" : pl,
            "functional_id": fid,
            "n_part_numbers": len(segments),
            "base_demand"  : round(base, 1),
            "trend_per_month": trend,
        })

        for seg_i, (seg_start_i, seg_end_i, pn) in enumerate(segments):
            seg_dates  = dates[seg_start_i:seg_end_i]
            seg_demand = signal[seg_start_i:seg_end_i]

            # Lookup table row
            lookup_rows.append({
                "part_number"  : pn,
                "functional_id": fid,
                "product_line" : pl,
                "active_from"  : seg_dates[0].strftime("%Y-%m-%d"),
                "active_to"    : "current" if seg_i == len(segments) - 1
                                           else seg_dates[-1].strftime("%Y-%m-%d"),
            })

            for d, qty in zip(seg_dates, seg_demand):
                demand_rows.append({
                    "date"         : d.strftime("%Y-%m-%d"),
                    "part_number"  : pn,
                    "functional_id": fid,
                    "product_line" : pl,
                    "demand"       : max(0, int(round(qty))),
                    "is_active"    : 1,
                })

            # Substitution event (recorded at the *start* of the new segment)
            if seg_i > 0:
                old_pn = segments[seg_i - 1][2]
                reason = rng.choice(SUBSTITUTION_REASONS)
                substitution_rows.append({
                    "event_date"      : seg_dates[0].strftime("%Y-%m-%d"),
                    "functional_id"   : fid,
                    "product_line"    : pl,
                    "old_part_number" : old_pn,
                    "new_part_number" : pn,
                    "reason"          : reason,
                })

# ─────────────────────────────────────────────
# Assemble DataFrames
# ─────────────────────────────────────────────
df_demand = pd.DataFrame(demand_rows)
df_demand["date"] = pd.to_datetime(df_demand["date"])
df_demand = df_demand.sort_values(["product_line", "functional_id", "date"]).reset_index(drop=True)

df_lookup = pd.DataFrame(lookup_rows)
df_subs   = pd.DataFrame(substitution_rows)
df_bom    = pd.DataFrame(bom_rows)

# ─────────────────────────────────────────────
# Write outputs
# ─────────────────────────────────────────────
os.makedirs("data", exist_ok=True)

df_demand.to_csv("data/demand_history.csv", index=False)
df_lookup.to_csv("data/functional_id_lookup.csv", index=False)
df_subs.to_csv("data/substitution_events.csv", index=False)
df_bom.to_csv("data/product_bom.csv", index=False)

# ─────────────────────────────────────────────
# Summary report
# ─────────────────────────────────────────────
print("=" * 60)
print("BOM-Aware Demand Forecasting — Synthetic Dataset Summary")
print("=" * 60)
print(f"\nDate range          : {dates[0].strftime('%b %Y')} – {dates[-1].strftime('%b %Y')} ({N_MONTHS} months)")
print(f"Product lines       : {N_PRODUCT_LINES}  ({', '.join(PRODUCT_LINES)})")
print(f"Functional families : {N_PRODUCT_LINES * N_COMPONENTS_EACH}")
print(f"Unique part numbers : {df_lookup['part_number'].nunique()}")
print(f"Substitution events : {len(df_subs)}")
print(f"Total demand rows   : {len(df_demand):,}")

print("\n── Substitution events per product line ──")
print(df_subs.groupby("product_line").size().to_string())

print("\n── Substitution reasons (all lines) ──")
print(df_subs["reason"].value_counts().to_string())

print("\n── Demand stats (units/month, across all active rows) ──")
print(df_demand["demand"].describe().round(1).to_string())

print("\n── Files written ──")
for f in ["data/demand_history.csv", "data/functional_id_lookup.csv",
          "data/substitution_events.csv", "data/product_bom.csv"]:
    size_kb = os.path.getsize(f) / 1024
    print(f"  {f:<40}  ({size_kb:.1f} KB)")

print("\nDone. ✓")
