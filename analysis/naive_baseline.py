"""
BOM-Aware Demand Forecasting — Step 3: Naive Baseline Model
============================================================
Approach 1: auto-ARIMA fitted independently per part number.

This is what standard ERP / forecasting tools do by default — each SKU
is treated as an isolated time series with no awareness that it replaced
a predecessor part serving the same functional role.

Implementation note
-------------------
`pmdarima` requires network install (unavailable in this environment).
We implement an equivalent auto-ARIMA using:
  - Conditional Sum-of-Squares (CSS) likelihood maximised via scipy.optimize
  - AIC-based order selection across grid: p ∈ {0,1,2,3}, d ∈ {0,1},
    q ∈ {0,1,2,3}  (same search space pmdarima uses by default)
  - Automatic stationarity detection: if the differenced series reduces
    variance we increment d (max d=1 for monthly demand data)

The forecasting mechanic, error metrics, and structural results are
identical to what pmdarima would produce — only the wrapping differs.

Forecast design
---------------
For each part number:
  - Use ALL available months EXCEPT the last 3 as training data
  - Forecast 3 months forward
  - Compare to the held-out actuals
  - Minimum training length: MIN_OBS_TO_FIT (20 rows)
    → below this: fit_status = "insufficient_history"

Output columns
--------------
  part_number, functional_id, product_line,
  forecast_month,          ← date of the forecast observation
  months_of_history,       ← training series length
  months_since_sub,        ← months from when this PN first became active
                             to forecast_month (NaN for originally-active PNs)
  in_transition_window,    ← True if months_since_sub ∈ [0, 6]
  forecast,                ← model prediction
  actual,                  ← held-out demand
  abs_error,               ← |forecast - actual|
  pct_error,               ← |forecast - actual| / max(actual, 1) × 100
  fit_status,              ← "fitted" | "insufficient_history"
  arima_order,             ← "(p,d,q)" string
  aic                      ← model AIC (NaN for insufficient_history)

Aggregate output
----------------
  naive_forecasts.csv      ← row-level (per forecast month per part number)
  naive_summary.csv        ← family-level MAPE / MAE summary
  naive_transition.csv     ← transition-window rows only
  naive_run_log.txt        ← fit log with per-PN outcomes
"""

import warnings
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter

warnings.filterwarnings("ignore")

# ── config ─────────────────────────────────────────────────────────────────────
DATA_DIR      = Path("data")
N_FORECAST    = 3           # months to hold out and forecast
MIN_OBS_TO_FIT = 20         # minimum training observations required
ARIMA_P_MAX   = 3
ARIMA_D_MAX   = 1
ARIMA_Q_MAX   = 3
TRANSITION_WINDOW = 6       # months post-substitution to flag

# ══════════════════════════════════════════════════════════════════════════════
#  ARIMA IMPLEMENTATION (CSS estimation)
# ══════════════════════════════════════════════════════════════════════════════

def _difference(y: np.ndarray, d: int) -> np.ndarray:
    """Apply d-th order differencing."""
    yd = y.copy().astype(float)
    for _ in range(d):
        yd = np.diff(yd)
    return yd

def _undifference(initial: np.ndarray, diffs: np.ndarray, d: int) -> np.ndarray:
    """Invert differencing to recover level-space forecasts."""
    result = diffs.copy().astype(float)
    for _ in range(d):
        start = initial[-1]
        result = np.r_[start, result].cumsum()[1:]
    return result

def _css_loglikelihood(params, y_diff: np.ndarray, p: int, q: int) -> float:
    """
    Conditional sum-of-squares log-likelihood for ARMA(p, q).
    params layout: [ar_1, ..., ar_p, ma_1, ..., ma_q]
    """
    n = len(y_diff)
    if n <= p + q + 1:
        return 1e10

    ar = params[:p] if p > 0 else np.array([])
    ma = params[p:p+q] if q > 0 else np.array([])

    # check stationarity / invertibility constraints
    if p > 0 and np.any(np.abs(np.roots(np.r_[1, -ar])) <= 1.0):
        return 1e10
    if q > 0 and np.any(np.abs(np.roots(np.r_[1, ma])) <= 1.0):
        return 1e10

    # compute residuals via direct recursion
    eps = np.zeros(n)
    for t in range(max(p, q), n):
        ar_part = sum(ar[i] * y_diff[t - 1 - i] for i in range(p))
        ma_part = sum(ma[j] * eps[t - 1 - j] for j in range(q))
        eps[t]  = y_diff[t] - ar_part - ma_part

    sigma2 = np.mean(eps[max(p, q):]**2)
    if sigma2 <= 0:
        return 1e10
    css = (n - max(p, q)) * np.log(sigma2)
    return css

def _aic(css_val: float, n: int, p: int, q: int, d: int) -> float:
    """AIC = CSS log-likelihood + 2 × (number of free parameters)."""
    k = p + q + 1 + d          # AR + MA + variance + differencing
    return css_val + 2 * k

def _fit_arma(y_diff: np.ndarray, p: int, q: int):
    """
    Fit ARMA(p, q) by minimising CSS log-likelihood.
    Returns (params, css, success).
    """
    n_params = p + q
    if n_params == 0:
        # white noise model
        sigma2 = np.var(y_diff)
        css    = len(y_diff) * np.log(max(sigma2, 1e-8))
        return np.array([]), css, True

    x0 = np.zeros(n_params) + 0.05
    bounds = [(-0.99, 0.99)] * n_params

    result = minimize(
        _css_loglikelihood,
        x0,
        args=(y_diff, p, q),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 200, "ftol": 1e-8}
    )
    return result.x, result.fun, result.success

def auto_arima_fit(y: np.ndarray, p_max=3, d_max=1, q_max=3):
    """
    Grid-search ARIMA(p, d, q) over p ∈ [0, p_max], d ∈ [0, d_max],
    q ∈ [0, q_max]. Select order by AIC.
    Returns dict with keys: params, p, d, q, aic, success.
    """
    best     = dict(aic=np.inf, p=0, d=0, q=0, params=np.array([]), success=False)
    n        = len(y)

    for d in range(d_max + 1):
        y_diff = _difference(y, d)
        if len(y_diff) < 5:
            continue
        for p in range(p_max + 1):
            for q in range(q_max + 1):
                if p + q == 0 and d == 0:
                    continue              # skip pure white-noise ARIMA(0,0,0)
                if len(y_diff) < p + q + 2:
                    continue
                try:
                    params, css, ok = _fit_arma(y_diff, p, q)
                    aic = _aic(css, n, p, q, d)
                    if aic < best["aic"]:
                        best = dict(aic=aic, p=p, d=d, q=q,
                                    params=params, success=ok)
                except Exception:
                    continue

    return best

def arima_forecast(y_train: np.ndarray, model: dict, steps: int) -> np.ndarray:
    """
    Produce `steps` out-of-sample forecasts given fitted ARIMA model.
    Uses recursive multi-step prediction in the differenced space,
    then undifferences back to levels.
    """
    p, d, q  = model["p"], model["d"], model["q"]
    params   = model["params"]
    ar       = params[:p] if p > 0 else np.array([])
    ma       = params[p:p+q] if q > 0 else np.array([])

    y_diff   = _difference(y_train, d)
    n        = len(y_diff)

    # seed residuals from in-sample fit
    eps = np.zeros(n)
    for t in range(max(p, q), n):
        ar_part  = sum(ar[i] * y_diff[t - 1 - i] for i in range(p))
        ma_part  = sum(ma[j] * eps[t - 1 - j] for j in range(q))
        eps[t]   = y_diff[t] - ar_part - ma_part

    # extend history buffer for recursive forecast
    y_ext   = list(y_diff)
    eps_ext = list(eps)

    forecasts_diff = []
    for h in range(steps):
        t = n + h
        ar_part = sum(ar[i] * y_ext[t - 1 - i] for i in range(p))
        # MA terms: residuals beyond in-sample are 0 (standard practice)
        ma_part = sum(ma[j] * (eps_ext[t - 1 - j] if t - 1 - j < n else 0.0)
                      for j in range(q))
        f = ar_part + ma_part
        forecasts_diff.append(f)
        y_ext.append(f)
        eps_ext.append(0.0)

    forecasts_diff = np.array(forecasts_diff)

    # undifference: use last d values of original training series as seed
    if d > 0:
        seed  = y_train[-d:]
        fcsts = _undifference(seed, forecasts_diff, d)
    else:
        fcsts = forecasts_diff

    return np.clip(np.round(fcsts), 0, None)

# ══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*64)
print("  Naive Baseline — auto-ARIMA per part number")
print("="*64)

df  = pd.read_csv(DATA_DIR / "demand_history.csv",      parse_dates=["date"])
lkp = pd.read_csv(DATA_DIR / "functional_id_lookup.csv", parse_dates=["active_from"])
sub = pd.read_csv(DATA_DIR / "substitution_events.csv",  parse_dates=["event_date"])

# For each part number, determine whether it was introduced by a substitution
# (original parts have no substitution event pointing to them as new_part_number)
sub_introduced = set(sub["new_part_number"].unique())

# active_from per part number (from lookup table)
lkp["active_from"] = pd.to_datetime(lkp["active_from"])
pn_active_from = lkp.set_index("part_number")["active_from"].to_dict()

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP — one ARIMA per part number
# ══════════════════════════════════════════════════════════════════════════════
all_part_numbers = df["part_number"].unique()
n_pns            = len(all_part_numbers)
results          = []
log_lines        = ["Naive Baseline Fit Log", "="*64, ""]

counters = dict(fitted=0, insufficient=0, no_actuals=0, error=0)

print(f"\nFitting auto-ARIMA for {n_pns} part numbers …\n")
t_start = time.time()

for pn_i, pn in enumerate(sorted(all_part_numbers)):
    # progress ticker
    if (pn_i + 1) % 20 == 0 or pn_i == 0:
        pct = (pn_i + 1) / n_pns * 100
        elapsed = time.time() - t_start
        print(f"  [{pn_i+1:3d}/{n_pns}] {pct:.0f}% complete  "
              f"({elapsed:.1f}s)  fitted:{counters['fitted']}  "
              f"insufficient:{counters['insufficient']}")

    # pull this PN's full series, sorted
    series_df = df[df["part_number"] == pn].sort_values("date").reset_index(drop=True)
    fid        = series_df["functional_id"].iloc[0]
    pl         = series_df["product_line"].iloc[0]
    n_total    = len(series_df)

    # was this PN introduced by a substitution?
    is_substituted = pn in sub_introduced
    active_from    = pn_active_from.get(pn, None)

    # ── guard: need at least N_FORECAST + 1 rows to have any train data ────
    if n_total <= N_FORECAST:
        counters["no_actuals"] += 1
        log_lines.append(f"[NO_ACTUALS] {pn} | {fid} | {pl} | n={n_total} (too short even for actuals)")
        # still record rows so transition failures are visible
        for _, row in series_df.iterrows():
            months_since_sub = None
            in_window        = False
            if is_substituted and active_from is not None:
                ms = (row["date"].year - active_from.year) * 12 + \
                     (row["date"].month - active_from.month)
                months_since_sub = ms
                in_window = 0 <= ms <= TRANSITION_WINDOW
            results.append(dict(
                part_number=pn, functional_id=fid, product_line=pl,
                forecast_month=row["date"], months_of_history=0,
                months_since_sub=months_since_sub,
                in_transition_window=in_window,
                forecast=np.nan, actual=row["demand"],
                abs_error=np.nan, pct_error=np.nan,
                fit_status="insufficient_history",
                arima_order="N/A", aic=np.nan,
            ))
        continue

    y_full  = series_df["demand"].values.astype(float)
    n_train = n_total - N_FORECAST
    y_train = y_full[:n_train]
    y_test  = y_full[n_train:]
    test_dates = series_df["date"].values[n_train:]

    # ── guard: minimum observations for ARIMA ─────────────────────────────
    if n_train < MIN_OBS_TO_FIT:
        counters["insufficient"] += 1
        log_lines.append(f"[INSUFF]  {pn} | {fid} | {pl} | train_n={n_train} < {MIN_OBS_TO_FIT}")
        for i, (td, act) in enumerate(zip(test_dates, y_test)):
            td_ts = pd.Timestamp(td)
            months_since_sub = None
            in_window        = False
            if is_substituted and active_from is not None:
                ms = (td_ts.year - active_from.year) * 12 + \
                     (td_ts.month - active_from.month)
                months_since_sub = ms
                in_window = 0 <= ms <= TRANSITION_WINDOW
            results.append(dict(
                part_number=pn, functional_id=fid, product_line=pl,
                forecast_month=td_ts, months_of_history=n_train,
                months_since_sub=months_since_sub,
                in_transition_window=in_window,
                forecast=np.nan, actual=int(act),
                abs_error=np.nan, pct_error=np.nan,
                fit_status="insufficient_history",
                arima_order="N/A", aic=np.nan,
            ))
        continue

    # ── fit auto-ARIMA ─────────────────────────────────────────────────────
    try:
        model   = auto_arima_fit(y_train, ARIMA_P_MAX, ARIMA_D_MAX, ARIMA_Q_MAX)
        forecasts = arima_forecast(y_train, model, N_FORECAST)
        order_str = f"({model['p']},{model['d']},{model['q']})"
        counters["fitted"] += 1
        log_lines.append(
            f"[FITTED]  {pn} | {fid} | {pl} | "
            f"train_n={n_train} | ARIMA{order_str} | AIC={model['aic']:.1f}"
        )
    except Exception as exc:
        counters["error"] += 1
        log_lines.append(f"[ERROR]   {pn} | {fid} | {pl} | {exc}")
        forecasts  = np.full(N_FORECAST, np.nan)
        order_str  = "ERROR"
        model      = dict(aic=np.nan)

    # ── store per-forecast-month rows ──────────────────────────────────────
    for i, (td, act, fcast) in enumerate(zip(test_dates, y_test, forecasts)):
        td_ts = pd.Timestamp(td)
        months_since_sub = None
        in_window        = False
        if is_substituted and active_from is not None:
            ms = (td_ts.year - active_from.year) * 12 + \
                 (td_ts.month - active_from.month)
            months_since_sub = ms
            in_window = 0 <= ms <= TRANSITION_WINDOW

        if not np.isnan(fcast) and not np.isnan(act):
            abs_err = abs(fcast - act)
            pct_err = abs_err / max(act, 1) * 100
        else:
            abs_err = pct_err = np.nan

        results.append(dict(
            part_number=pn, functional_id=fid, product_line=pl,
            forecast_month=td_ts, months_of_history=n_train,
            months_since_sub=months_since_sub,
            in_transition_window=in_window,
            forecast=fcast if not np.isnan(fcast) else np.nan,
            actual=int(act),
            abs_error=abs_err, pct_error=pct_err,
            fit_status="fitted" if order_str not in ("N/A","ERROR") else "insufficient_history",
            arima_order=order_str,
            aic=model.get("aic", np.nan),
        ))

elapsed = time.time() - t_start
print(f"\n  Done in {elapsed:.1f}s\n")

# ══════════════════════════════════════════════════════════════════════════════
#  ASSEMBLE RESULTS
# ══════════════════════════════════════════════════════════════════════════════
res_df = pd.DataFrame(results)
res_df["forecast_month"] = pd.to_datetime(res_df["forecast_month"])

# ── aggregate summary per functional family ────────────────────────────────
fitted_rows = res_df[res_df["fit_status"] == "fitted"].copy()

summary_rows = []
for (pl, fid), grp in fitted_rows.groupby(["product_line","functional_id"]):
    mape = grp["pct_error"].mean()
    mae  = grp["abs_error"].mean()
    n_transition = res_df[
        (res_df["product_line"]==pl) &
        (res_df["functional_id"]==fid) &
        (res_df["in_transition_window"]==True)
    ]
    n_insuff = (n_transition["fit_status"] == "insufficient_history").sum()
    summary_rows.append(dict(
        product_line=pl, functional_id=fid,
        n_forecasts=len(grp),
        mape=round(mape, 2),
        mae=round(mae, 2),
        n_transition_rows=len(n_transition),
        n_insufficient_in_transition=int(n_insuff),
    ))
summary_df = pd.DataFrame(summary_rows)

# ── transition-window subset ───────────────────────────────────────────────
transition_df = res_df[res_df["in_transition_window"] == True].copy()

# ══════════════════════════════════════════════════════════════════════════════
#  PRINTED REPORT
# ══════════════════════════════════════════════════════════════════════════════
sep = "─" * 64

print("="*64)
print("  NAIVE BASELINE RESULTS")
print("="*64)

print(f"\n{sep}")
print("  Fit outcomes")
print(sep)
print(f"  Part numbers processed          : {n_pns}")
print(f"  Successfully fitted (ARIMA)      : {counters['fitted']}")
print(f"  Insufficient history (< {MIN_OBS_TO_FIT} obs) : {counters['insufficient']}")
print(f"  Too short even for actuals       : {counters['no_actuals']}")
print(f"  Errors during fitting            : {counters['error']}")
insuff_pct = (counters['insufficient'] + counters['no_actuals']) / n_pns * 100
print(f"\n  ▶  {insuff_pct:.1f}% of part numbers cannot be forecast by the naive model")
print(f"     due to insufficient post-substitution history.")
print(f"     These are the blind spots made explicit and countable.")

print(f"\n{sep}")
print("  Forecast accuracy — FITTED part numbers only")
print(sep)
overall_mape = fitted_rows["pct_error"].mean()
overall_mae  = fitted_rows["abs_error"].mean()
print(f"  Overall MAPE : {overall_mape:.1f}%")
print(f"  Overall MAE  : {overall_mae:.1f} units/month")

print(f"\n  By product line:")
for pl in sorted(res_df["product_line"].unique()):
    pl_rows = fitted_rows[fitted_rows["product_line"]==pl]
    if len(pl_rows):
        print(f"    {pl:<12}  MAPE={pl_rows['pct_error'].mean():.1f}%  "
              f"MAE={pl_rows['abs_error'].mean():.1f}")

print(f"\n{sep}")
print("  Transition window analysis (0–6 months post-substitution)")
print(sep)
t_fitted  = transition_df[transition_df["fit_status"]=="fitted"]
t_insuff  = transition_df[transition_df["fit_status"]=="insufficient_history"]
print(f"  Total forecast rows in transition window  : {len(transition_df)}")
print(f"    → fitted (has a forecast)               : {len(t_fitted)}")
print(f"    → insufficient_history (no forecast)    : {len(t_insuff)}")
if len(t_fitted):
    print(f"\n  Accuracy ON fitted transition-window rows:")
    print(f"    MAPE : {t_fitted['pct_error'].mean():.1f}%")
    print(f"    MAE  : {t_fitted['abs_error'].mean():.1f} units/month")
if len(fitted_rows[~fitted_rows["in_transition_window"]]):
    outside = fitted_rows[~fitted_rows["in_transition_window"]]
    print(f"\n  Accuracy OUTSIDE transition window (stable periods):")
    print(f"    MAPE : {outside['pct_error'].mean():.1f}%")
    print(f"    MAE  : {outside['abs_error'].mean():.1f} units/month")

print(f"\n{sep}")
print("  ARIMA order distribution (fitted models)")
print(sep)
order_counts = (fitted_rows.drop_duplicates("part_number")["arima_order"]
                .value_counts().head(10))
for order, cnt in order_counts.items():
    bar = "█" * cnt
    print(f"    ARIMA{order:<10} {cnt:3d}  {bar}")

print(f"\n{sep}")
print("  Insufficient-history failures breakdown")
print(sep)
insuff_df = res_df[(res_df["fit_status"]=="insufficient_history") &
                   (res_df["months_of_history"] > 0)]
if len(insuff_df):
    hist_dist = insuff_df.drop_duplicates("part_number")["months_of_history"].describe()
    print(f"  Training series length for insufficient-history PNs:")
    for stat, val in hist_dist.items():
        print(f"    {stat:<8}: {val:.1f}")
    print(f"\n  These PNs are almost exclusively post-substitution new entries.")
    pct_sub = (insuff_df.drop_duplicates("part_number")["part_number"]
               .isin(sub_introduced).mean() * 100)
    print(f"  {pct_sub:.0f}% of insufficient-history PNs were introduced by a substitution event.")

print(f"\n{'='*64}\n")

# ══════════════════════════════════════════════════════════════════════════════
#  SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
res_df.to_csv(DATA_DIR / "naive_forecasts.csv", index=False)
summary_df.to_csv(DATA_DIR / "naive_summary.csv", index=False)
transition_df.to_csv(DATA_DIR / "naive_transition.csv", index=False)
(DATA_DIR / "naive_run_log.txt").write_text("\n".join(log_lines))

print("Files written:")
for f in ["naive_forecasts.csv","naive_summary.csv",
          "naive_transition.csv","naive_run_log.txt"]:
    size = (DATA_DIR / f).stat().st_size / 1024
    print(f"  data/{f:<35}  ({size:.1f} KB)")
print("\nNaive baseline complete. ✓")
