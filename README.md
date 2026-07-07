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
