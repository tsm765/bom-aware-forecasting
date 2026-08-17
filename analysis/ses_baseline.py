"""
BOM-Aware Demand Forecasting — Step 3b: SES Naive Baseline
===========================================================
Simple Exponential Smoothing (SES) fitted independently per part number.

SES is the primary naive baseline because:
  - It fits ANY series length ≥ 1 observation — no minimum history gate
  - It's the workhorse of ERP demand planning at scale (SAP, Oracle, etc.)
  - It adapts to level shifts, making it the natural "best effort" a planner
    would use for a newly introduced part number with only a few months data
  - It makes the comparison honest: we're not comparing "ARIMA on long series"
    vs "BOM-aware on chained series" — we're comparing the SAME smoother
    applied to isolated-PN history vs chained functional-ID history

SES formula:
    S_t = α * y_t + (1 - α) * S_{t-1}
    Forecast for all h steps ahead: ŷ_{t+h} = S_t   (flat extrapolation)

Alpha selection: minimise one-step-ahead MSE on training series via
    Brent scalar minimisation over α ∈ [0.05, 0.95]
    (identical to what statsmodels SimpleExpSmoothing does)

Minimum series length: 1 observation (α = 0.5 default for single-obs series)

Output schema (identical to naive_baseline.py / Step 3)
--------------------------------------------------------
  part_number, functional_id, product_line,
  forecast_month, months_of_history,
  months_since_sub, in_transition_window,
  forecast, actual, abs_error, pct_error,
  fit_status,      ← "fitted" for all SES (no failures expected)
  alpha,           ← optimised smoothing parameter
  model_type       ← "SES"

Files written
-------------
  data/ses_forecasts.csv
  data/ses_summary.csv
  data/ses_transition.csv
"""

import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

warnings.filterwarnings("ignore")

DATA_DIR          = Path("data")
N_FORECAST        = 3
TRANSITION_WINDOW = 6
DEFAULT_ALPHA     = 0.5    # used when series has only 1 obs

# ══════════════════════════════════════════════════════════════════════════════
#  SES IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def _ses_mse(alpha: float, y: np.ndarray) -> float:
    """One-step-ahead MSE for a given alpha over training series y."""
    s = y[0]
    sse = 0.0
    for t in range(1, len(y)):
        err  = y[t] - s
        sse += err ** 2
        s    = alpha * y[t] + (1 - alpha) * s
    return sse / max(len(y) - 1, 1)

def ses_fit(y_train: np.ndarray) -> dict:
    """
    Fit SES on y_train. Returns dict with alpha, final_level, mse.
    Handles series of any length ≥ 1.
    """
    n = len(y_train)
    if n == 1:
        return dict(alpha=DEFAULT_ALPHA, level=float(y_train[0]), mse=np.nan)

    result = minimize_scalar(
        _ses_mse,
        args=(y_train,),
        bounds=(0.05, 0.95),
        method="bounded",
        options={"xatol": 1e-6}
    )
    alpha = result.x

    # compute final level
    s = y_train[0]
    for t in range(1, n):
        s = alpha * y_train[t] + (1 - alpha) * s

    return dict(alpha=alpha, level=s, mse=result.fun)

def ses_forecast(model: dict, steps: int) -> np.ndarray:
    """SES flat forecast: all h-step forecasts = final smoothed level."""
    return np.clip(np.round(np.full(steps, model["level"])), 0, None)

# ══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*64)
print("  SES Naive Baseline — per part number (Step 3b)")
print("="*64)

df  = pd.read_csv(DATA_DIR / "demand_history.csv",       parse_dates=["date"])
lkp = pd.read_csv(DATA_DIR / "functional_id_lookup.csv", parse_dates=["active_from"])
sub = pd.read_csv(DATA_DIR / "substitution_events.csv",  parse_dates=["event_date"])

sub_introduced = set(sub["new_part_number"].unique())
lkp["active_from"] = pd.to_datetime(lkp["active_from"])
pn_active_from = lkp.set_index("part_number")["active_from"].to_dict()

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════
all_pns = sorted(df["part_number"].unique())
n_pns   = len(all_pns)
results = []

print(f"\nFitting SES for {n_pns} part numbers …")
t_start = time.time()

for pn_i, pn in enumerate(all_pns):
    series_df = df[df["part_number"] == pn].sort_values("date").reset_index(drop=True)
    fid   = series_df["functional_id"].iloc[0]
    pl    = series_df["product_line"].iloc[0]
    n_tot = len(series_df)

    is_substituted = pn in sub_introduced
    active_from    = pn_active_from.get(pn, None)

    # ── split train / test ─────────────────────────────────────────────────
    # If the series is shorter than N_FORECAST + 1, use all but last obs as
    # train and last obs as the single test point (SES needs ≥1 train obs)
    if n_tot <= N_FORECAST:
        n_train = max(1, n_tot - 1)
    else:
        n_train = n_tot - N_FORECAST

    y_full     = series_df["demand"].values.astype(float)
    y_train    = y_full[:n_train]
    y_test     = y_full[n_train:]
    test_dates = series_df["date"].values[n_train:]
    n_steps    = len(y_test)

    # ── fit & forecast ─────────────────────────────────────────────────────
    model     = ses_fit(y_train)
    forecasts = ses_forecast(model, n_steps)

    # ── store rows ─────────────────────────────────────────────────────────
    for i, (td, act, fcast) in enumerate(zip(test_dates, y_test, forecasts)):
        td_ts = pd.Timestamp(td)
        months_since_sub = None
        in_window        = False
        if is_substituted and active_from is not None:
            ms = (td_ts.year - active_from.year) * 12 + \
                 (td_ts.month - active_from.month)
            months_since_sub = ms
            in_window = 0 <= ms <= TRANSITION_WINDOW

        abs_err = abs(fcast - act)
        pct_err = abs_err / max(act, 1) * 100

        results.append(dict(
            part_number=pn, functional_id=fid, product_line=pl,
            forecast_month=td_ts,
            months_of_history=int(n_train),
            months_since_sub=months_since_sub,
            in_transition_window=in_window,
            forecast=fcast, actual=int(act),
            abs_error=abs_err, pct_error=pct_err,
            fit_status="fitted",
            alpha=round(model["alpha"], 4),
            model_type="SES",
        ))

elapsed = time.time() - t_start
print(f"  Done in {elapsed:.2f}s — all {n_pns} part numbers fitted.\n")

# ══════════════════════════════════════════════════════════════════════════════
#  ASSEMBLE & REPORT
# ══════════════════════════════════════════════════════════════════════════════
res_df = pd.DataFrame(results)
res_df["forecast_month"] = pd.to_datetime(res_df["forecast_month"])

sep = "─" * 64

print("="*64)
print("  SES BASELINE RESULTS")
print("="*64)

print(f"\n{sep}")
print("  Fit outcomes")
print(sep)
print(f"  Part numbers processed : {n_pns}")
print(f"  Successfully fitted    : {n_pns}   (100% — SES has no minimum history gate)")
print(f"  Insufficient history   : 0")

print(f"\n{sep}")
print("  Smoothing parameter α distribution")
print(sep)
alpha_vals = res_df.drop_duplicates("part_number")["alpha"]
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
labels = ["0.05–0.20","0.20–0.40","0.40–0.60","0.60–0.80","0.80–0.95"]
alpha_hist = pd.cut(alpha_vals, bins=bins, labels=labels).value_counts().sort_index()
for label, cnt in alpha_hist.items():
    bar = "█" * cnt
    print(f"  α  {label}   {cnt:3d}  {bar}")
high_alpha = (alpha_vals > 0.6).mean() * 100
print(f"\n  {high_alpha:.0f}% of part numbers have α > 0.6 — model leans heavily on")
print(f"  recent observations. Expected: short series + noisy demand.")

print(f"\n{sep}")
print("  Forecast accuracy — all part numbers")
print(sep)
overall_mape = res_df["pct_error"].mean()
overall_mae  = res_df["abs_error"].mean()
print(f"  Overall MAPE : {overall_mape:.1f}%")
print(f"  Overall MAE  : {overall_mae:.1f} units/month")

print(f"\n  By product line:")
for pl in sorted(res_df["product_line"].unique()):
    pl_rows = res_df[res_df["product_line"]==pl]
    print(f"    {pl:<12}  MAPE={pl_rows['pct_error'].mean():.1f}%  "
          f"MAE={pl_rows['abs_error'].mean():.1f}")

print(f"\n{sep}")
print("  Transition window analysis (0–6 months post-substitution)")
print(sep)
trans_df = res_df[res_df["in_transition_window"] == True]
print(f"  Transition-window rows       : {len(trans_df)}")
print(f"  Fitted (has forecast)        : {len(trans_df)}  ← 100% coverage vs ARIMA 0%")
if len(trans_df):
    print(f"  Transition-window MAPE       : {trans_df['pct_error'].mean():.1f}%")
    print(f"  Transition-window MAE        : {trans_df['abs_error'].mean():.1f} units/month")
outside = res_df[~res_df["in_transition_window"]]
print(f"\n  Stable-period MAPE           : {outside['pct_error'].mean():.1f}%")
print(f"  Stable-period MAE            : {outside['abs_error'].mean():.1f} units/month")
delta_mape = trans_df["pct_error"].mean() - outside["pct_error"].mean()
print(f"\n  ▶  Transition-window MAPE penalty vs stable periods: +{delta_mape:.1f}pp")
print(f"     This is the quantified cost of history blindness in the naive model.")

# months_since_sub breakdown
print(f"\n  Accuracy by months-since-substitution:")
for ms in range(TRANSITION_WINDOW + 1):
    ms_rows = trans_df[trans_df["months_since_sub"] == ms]
    if len(ms_rows):
        print(f"    Month +{ms}   n={len(ms_rows):2d}  "
              f"MAPE={ms_rows['pct_error'].mean():.1f}%  "
              f"MAE={ms_rows['abs_error'].mean():.1f}")

print(f"\n{'='*64}\n")

# ══════════════════════════════════════════════════════════════════════════════
#  SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
summary_rows = []
for (pl, fid), grp in res_df.groupby(["product_line", "functional_id"]):
    t_rows = grp[grp["in_transition_window"]==True]
    summary_rows.append(dict(
        product_line=pl, functional_id=fid,
        n_forecasts=len(grp),
        mape=round(grp["pct_error"].mean(), 2),
        mae=round(grp["abs_error"].mean(), 2),
        n_transition_rows=len(t_rows),
        transition_mape=round(t_rows["pct_error"].mean(), 2) if len(t_rows) else np.nan,
        transition_mae=round(t_rows["abs_error"].mean(), 2)  if len(t_rows) else np.nan,
    ))

summary_df    = pd.DataFrame(summary_rows)
transition_df = res_df[res_df["in_transition_window"]==True].copy()

res_df.to_csv(DATA_DIR / "ses_forecasts.csv", index=False)
summary_df.to_csv(DATA_DIR / "ses_summary.csv", index=False)
transition_df.to_csv(DATA_DIR / "ses_transition.csv", index=False)

print("Files written:")
for f in ["ses_forecasts.csv", "ses_summary.csv", "ses_transition.csv"]:
    size = (DATA_DIR / f).stat().st_size / 1024
    print(f"  data/{f:<35} ({size:.1f} KB)")
print("\nSES baseline complete. ✓")
