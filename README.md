# backtest-overfitting

[![tests](https://github.com/jarvisss007/backtest-overfitting/actions/workflows/ci.yml/badge.svg)](https://github.com/jarvisss007/backtest-overfitting/actions/workflows/ci.yml)

Tools for detecting when a backtest is fooling you.

Trying many strategy configurations and reporting the best one's Sharpe is the most common way
to produce a spurious "edge" — the winner is often just the luckiest of many coin-flips. This
library quantifies how much of an apparent edge is an artifact of selection and multiple testing,
using the standard methods from the Bailey / López de Prado literature.

```python
import numpy as np
from overfit import analyze, format_report

returns = np.random.normal(0, 0.01, size=(1000, 200))   # 200 strategies × 1000 days, no real edge
print(format_report(analyze(returns)))
# VERDICT: OVERFIT: not significant after deflation AND high overfitting probability
```

## What's implemented

| Function | Question it answers |
|---|---|
| `probabilistic_sharpe_ratio` | Is a Sharpe significantly above a benchmark, given the returns' skew & kurtosis and the sample length? |
| `expected_max_sharpe` | What Sharpe would I expect from the **best of N** random trials with no true edge? |
| `deflated_sharpe_ratio` (DSR) | Is the Sharpe significant *after* deflating for how many strategies I tried? |
| `min_track_record_length` | How long must a track record be before I can trust a Sharpe? |
| `min_backtest_length` | How many years of backtest do N trials require before a given Sharpe means anything? |
| `pbo_cscv` | **Probability of Backtest Overfitting** via combinatorially symmetric cross-validation. |
| `analyze` / `format_report` | One-call workup on a `(T × N)` matrix of strategy returns. |
| `monte_carlo_drawdown` | If this edge is real, how bad could the ride actually get? Shuffles the same trades into thousands of orderings and reports the max-drawdown distribution — see [case study #4](case_studies/turn-of-month-drawdown.md). |

## Why it works — the demo

`python demo.py` runs two scenarios and the min-backtest-length table:

- **Scenario A — data mining.** 200 pure-noise strategies; the best shows an annualized Sharpe
  of **1.34** (looks publishable). The tools expose it: Deflated Sharpe **0.36**, PBO **0.50**,
  negative IS→OOS degradation → **OVERFIT**.
- **Scenario B — a strong real edge** (Sharpe ~3) hidden among 199 noise strategies. It's
  selected and **survives**: DSR **0.998**, PBO **0.04**. (A *modest* real edge of ~1.3 gets
  drowned by 200 noise trials — which is itself the honest lesson.)
- **Min backtest length:** try 1000 configurations and an annualized Sharpe of 1.0 needs ~10
  years of data before it means anything.

A tool you can't trust is worse than none, so the test suite asserts both directions: it must
flag cherry-picked noise **and** let a genuine edge through.

## Case studies: my own research

[`case_studies/zero-dte-audit.md`](case_studies/zero-dte-audit.md) — I ran the toolkit against
a 6-config strategy sweep from my own 0DTE research. With no code inspection, "pick the best
in-sample Sharpe" selected exactly the config containing a lookahead bug, and the tools
convicted it (DSR 0.34, PBO 0.60, IS→OOS slope −1.44 → **OVERFIT**) — reproducing in seconds
a verdict that originally took a manual bug-hunt.

[`case_studies/daily-meanrev.md`](case_studies/daily-meanrev.md) — the mirror image: SPY
daily mean reversion is a **real** anomaly (18 trials, PBO 0.002, every long variant
profitable after costs) that still fails as a strategy (DSR 0.59; walk-forward Sharpe 0.26
vs buy-and-hold 0.53). Real-but-weak edges produce exactly this PBO≈0 / DSR<0.95 signature —
the two metrics measure different things, and you need both. Fully reproducible from public
data: `python case_studies/daily_meanrev_repro.py`.

[`case_studies/insider-gap.md`](case_studies/insider-gap.md) — the third failure mode:
an **execution assumption**. SEC Form 4 insider purchases show t≈3 abnormal returns and a
net Sharpe of +3.1 entering at the filing-day close — but filings mostly arrive after hours,
and at the honest entry (next day's open) the edge vanishes entirely (all |t|<2, every
variant net-negative → OVERFIT). The whole effect lives in the untradeable overnight gap.
Code bug (study #1), weak edge (study #2), false execution assumption (study #3) — three
different ways a good-looking backtest lies.

## Install & run

```
pip install -r requirements.txt      # numpy, scipy
python demo.py                       # see it in action
pytest -q                            # 7 property/behaviour tests
```

Use it on your own research by passing a `(T observations × N strategies)` matrix of period
returns to `analyze(...)`.

## References

- Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*, Journal of Portfolio Management.
- Bailey, Borwein, López de Prado & Zhu (2015), *The Probability of Backtest Overfitting*,
  Journal of Computational Finance.
- Bailey, Borwein, López de Prado & Zhu (2014), *Pseudo-Mathematics and Financial Charlatanism*,
  Notices of the AMS.

MIT licensed.
