# BOM-Aware Demand Forecasting Under Component Substitution

A proof-of-concept showing that chaining demand histories across component substitutions
improves forecast accuracy during transition windows, where procurement decisions are
most exposed.

**Live dashboard:** https://bom-dashboard-kppefwmkxfqcsr4lidpwgq.streamlit.app/

Completed as a research project in Operations and Supply Network Analytics,
University of Oulu.

---

## The problem

Most forecasting systems are built around part numbers. When a component is substituted,
the successor part number starts with no demand history, no trend, and no seasonal
pattern. For the next few months the model is working with almost nothing, which is
exactly when procurement needs a reliable signal.

For manufacturers running monthly S&OP cycles this matters for a specific reason. The
material demand forecast that drives procurement comes from a supply plan, which comes
from a demand review, which is broken down through the BOM via MRP. That chain is only
as good as the demand signals at the bottom of it. When a substitution breaks the signal,
the error does not stay local.

## The approach

A functional ID is assigned to each component role. Predecessor and successor part
numbers serving the same function share that ID. Demand histories are chained under the
functional ID before the model is fitted, so the forecaster sees continuous functional
demand rather than a series that restarts at every engineering change.

The algorithm does not change. The ERP system does not change. The BOM structure does
not change. One mapping layer connects part number history to functional demand.

## Results

Two models were compared using an identical SES implementation, identical parameter
selection, and an identical train and test split. The only difference is the input
series: per part number for the naive model, chained by functional ID for the BOM-aware
model.

| Metric | Naive SES | BOM-aware SES |
|---|---|---|
| Overall MAPE | 14.5% | 11.2% |
| Overall MAE | 14.9 units/month | 12.6 units/month |
| Transition-window MAPE | 11.5% | 8.8% |
| Transition-window MAE | 12.8 units/month | 11.4 units/month |
| Fit rate | 100% | 100% |

The improvement holds across all four product lines. An internal consistency check
supports the mechanism: BOM-aware models settled on an average smoothing parameter of
0.30 against 0.40 for the naive models, meaning they lean more on historical pattern
than on recent observations, which is what a 33-month training series should produce
against an 8-month one.

### Why SES and not ARIMA

ARIMA was evaluated first and was rejected on structural grounds rather than accuracy.
It requires a minimum amount of history to fit reliably, and in the transition window
history is precisely what is missing. ARIMA fitted only 20 of 125 part numbers, and
produced no forecast at all for any of the 61 forecast months falling inside a
transition window. On the 20 series where it did fit, SES was slightly more accurate
anyway. SES has no minimum history requirement and reflects how most ERP systems handle
short-history components in practice, which makes it both the more honest and the more
demanding baseline.

## Cost translation

Forecast error converts into inventory cost through three channels: safety stock
holding, emergency ordering, and stockout exposure. Assumptions were chosen to be
defensible rather than impressive, and machine downtime is excluded entirely.

| Parameter | Value | Rationale |
|---|---|---|
| Service level | 95%, Z = 1.65 | Standard for industrial spare parts |
| Lead time | 6 weeks | Conservative for certified heavy equipment components |
| Holding cost rate | 22% per year | Standard operations management literature |
| Ordering cost | €100 per PO | Conservative; emergency procurement of certified parts typically costs more |
| Unit cost baseline | €200 | Sensitivity provided from €50 to €2,000 |
| Substitution events | 26 per year | 77 events observed over 36 months |
| σ proxy | MAE | Reasonable approximation for symmetric error distributions at this scale |

At the €200 baseline the modelled saving is **€161 per substitution event per year**.
Scaled linearly, 1,000 substitution events per year gives roughly **€161,000 annually**.

One point of clarity on that figure: the scaling unit is substitution *events*, not BOM
size. A portfolio of 10,000 components might generate only 100 substitution events in a
year. The dashboard slider is labelled by events for this reason.

## Dataset

Real BOM demand data is proprietary, so the dataset was designed rather than sourced:
four product lines, 48 component families, 125 part numbers, 36 months of monthly
demand, 77 substitution events.

Designing the data means encoding what is known about how these systems behave. Demand
for a function continues even when the part number serving it changes. January brings a
procurement surge driven by budget cycles. Substitution events cluster around generation
transitions. Building a dataset that reflects these patterns and then testing whether
the model recovers them is a more transparent form of validation than fitting to one
company's history.

A 13-check validation suite confirms the generated data behaves as designed: demand
continuity across substitutions, minimum six-month event spacing, detectable January
seasonality at a 25.2% premium, referential integrity, and no gaps or nulls.

## Repository structure

```
├── app.py                  Streamlit dashboard, single file
├── requirements.txt        Pinned dependencies
├── analysis/               Pipeline scripts, run in order
│   ├── generate_synthetic_data.py
│   ├── validate_data.py
│   ├── naive_baseline.py       ARIMA severity exhibit
│   ├── ses_baseline.py         Primary naive baseline
│   ├── compare_arima_ses.py    Methodology justification
│   ├── bom_aware_ses.py        BOM-aware model
│   └── cost_simulation.py      Cost translation
├── data/                   Generated dataset and model outputs
└── plots/                  Figures rendered by the dashboard
```

## Running it

```bash
git clone https://github.com/YOUR_USERNAME/bom-aware-forecasting.git
cd bom-aware-forecasting
pip install -r requirements.txt
streamlit run app.py
```

To regenerate everything from scratch:

```bash
cd analysis
python generate_synthetic_data.py
python validate_data.py
python naive_baseline.py
python ses_baseline.py
python compare_arima_ses.py
python bom_aware_ses.py
python cost_simulation.py
```

Note that `naive_baseline.py` implements auto-ARIMA directly using conditional
sum-of-squares estimation with AIC-based order selection, rather than importing
`pmdarima`. The estimation path is equivalent to what that library uses internally.

## Scope and limitations

The model assumes clean substitutions, where one part replaces another at a single point
in time. In practice there are often transition periods where both components are being
ordered simultaneously, which creates a demand-splitting effect and adds its own
procurement complexity. Handling that properly is the logical next step, and it would
likely increase the estimated saving rather than reduce it.

At five and six months post-substitution the MAE results are mixed. This is expected. As
time passes the naive model accumulates its own history and the gap narrows. The window
where this approach makes the biggest difference is the acute early period, the first
two to four months, which is also where procurement exposure is highest.

Results are from synthetic data. The claim is that the approach is sound and the
improvement is measurable, not that a specific saving has been demonstrated for any
particular company. Validation against real ERP data is the natural next step.

## A note on company references

Normet Group is referenced in the dashboard as a representative example of the class of
manufacturer this work addresses. All figures cited are drawn from publicly available
company information. No proprietary operational data from Normet or any other company
has been used.

## Author

Talha Malik
MSc Information Processing Science, Business Analytics
University of Oulu
