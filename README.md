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
| `monte_carlo_drawdown` | If this edge is real, how bad could the ride actually get? Shuffles the same trades into thousands of orderings and reports the max-drawdown distribution — see [case study #4](https://github.com/jarvisss007/backtest-overfitting/blob/main/case_studies/turn-of-month-drawdown.md). |

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

[`case_studies/zero-dte-audit.md`](https://github.com/jarvisss007/backtest-overfitting/blob/main/case_studies/zero-dte-audit.md) — I ran the toolkit against
a 6-config strategy sweep from my own 0DTE research. With no code inspection, "pick the best
in-sample Sharpe" selected exactly the config containing a lookahead bug, and the tools
convicted it (DSR 0.34, PBO 0.60, IS→OOS slope −1.44 → **OVERFIT**) — reproducing in seconds
a verdict that originally took a manual bug-hunt.

[`case_studies/daily-meanrev.md`](https://github.com/jarvisss007/backtest-overfitting/blob/main/case_studies/daily-meanrev.md) — the mirror image: SPY
daily mean reversion is a **real** anomaly (18 trials, PBO 0.002, every long variant
profitable after costs) that still fails as a strategy (DSR 0.59; walk-forward Sharpe 0.26
vs buy-and-hold 0.53). Real-but-weak edges produce exactly this PBO≈0 / DSR<0.95 signature —
the two metrics measure different things, and you need both. Fully reproducible from public
data: `python case_studies/daily_meanrev_repro.py`.

[`case_studies/insider-gap.md`](https://github.com/jarvisss007/backtest-overfitting/blob/main/case_studies/insider-gap.md) — the third failure mode:
an **execution assumption**. SEC Form 4 insider purchases show t≈3 abnormal returns and a
net Sharpe of +3.1 entering at the filing-day close — but filings mostly arrive after hours,
and at the honest entry (next day's open) the edge vanishes entirely (all |t|<2, every
variant net-negative → OVERFIT). The whole effect lives in the untradeable overnight gap.
[`case_studies/turn-of-month-drawdown.md`](https://github.com/jarvisss007/backtest-overfitting/blob/main/case_studies/turn-of-month-drawdown.md) — the
case where every gate **passes**: turn-of-month t5 scores DSR **1.0**, PBO **0.01**,
walk-forward 5/5 — the cleanest statistical profile in my knowledge base. DSR and PBO
answer "is this Sharpe above zero, and not an artifact of trying many configs?" They never
compare it with buy-and-hold, which this long-only basket still loses to: it beats zero, not
the market. They also say nothing about what living through it would feel like. Reshuffling the same 3,750 daily returns
5,000 times puts the *typical* max drawdown at 26.0% and the unlucky-but-ordinary p95 at
37.6% — the single historical path you observed is one draw from a much wider range.
Surviving the significance tests is not the same as being survivable.

[`case_studies/ml_model_search/results.md`](https://github.com/jarvisss007/backtest-overfitting/blob/main/case_studies/ml_model_search/results.md) — the
control experiment: a model search on a **null world** where nothing is predictable by
construction. 5 model families × 4 lookback windows (20 candidates, walk-forward) on a seeded
zero-mean price series; the best backtest Sharpe is **+0.84**, the same model out-of-sample is
**−0.54** (DSR **0.57**, PBO **0.55**, backtest 5.0 years vs 5.1 needed → **OVERFIT**). Repeated
on 30 fresh seeds, a positive in-sample "winner" appears every time and DSR rejects all of them;
PBO alone sits near 0.5, a coin flip, as it should on noise. Full search surface, figure and code
in the folder; the pre-registered design is [`DESIGN.md`](https://github.com/jarvisss007/backtest-overfitting/blob/main/case_studies/ml_model_search/DESIGN.md).

Code bug (study #1), weak edge (study #2), false execution assumption (study #3), a
strategy that beats zero but not the market, with drawdowns you may not sit through (study #4), and a search whose winner is pure
noise by construction (study #5) — five different ways a good-looking backtest misleads you.

## Install & run

```bash
pip install backtest-overfitting                                     # Python 3.10+, numpy and scipy
pip install git+https://github.com/jarvisss007/backtest-overfitting  # or the latest from GitHub
```

The import name is `overfit` (`pip install overfit` is someone else's project). From
Python, pass a `(T observations × N strategies)` matrix of period returns to `analyze(...)`, as in the example at the
top. Or skip the code: point the `backtest-overfitting` command at a CSV with one row per period, one column per
strategy and a header row of names. A leading date column is ignored, so `df.to_csv("returns.csv")` from pandas works
as it is.

```bash
backtest-overfitting returns.csv              # --periods-per-year 52 for weekly returns, --json for machine output
```

On 200 configurations of pure noise, four years of daily returns:

```
  Best strategy (#135) Sharpe (ann.):   1.34
  Probabilistic Sharpe (vs 0):                0.996
  Deflated Sharpe (vs best-of-N benchmark):   0.362   (benchmark 1.52 ann.)
  Prob. of Backtest Overfitting (PBO):        0.502   (>0.5 is bad)
  ...
  VERDICT: OVERFIT: not significant after deflation AND high overfitting probability
  Best strategy column: config_135
```

On its own, the winner looks 99.6% significant. Measured against the best of 200 tries, it isn't.

To work on the library itself:

```bash
git clone https://github.com/jarvisss007/backtest-overfitting && cd backtest-overfitting
pip install -e ".[test]"                     # adds pytest, scikit-learn and matplotlib (tests, case study 5)
python demo.py                               # see it in action
python case_studies/ml_model_search/run_search.py   # case study 5, ~2 minutes
pytest -q                                    # 29 property/behaviour tests
```

## References

- Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*, Journal of Portfolio Management.
- Bailey, Borwein, López de Prado & Zhu (2015), *The Probability of Backtest Overfitting*,
  Journal of Computational Finance.
- Bailey, Borwein, López de Prado & Zhu (2014), *Pseudo-Mathematics and Financial Charlatanism*,
  Notices of the AMS.

MIT licensed.
