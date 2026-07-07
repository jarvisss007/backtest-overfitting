# Case study: deflating my own 0DTE strategy sweep

The most useful test of an overfitting toolkit is pointing it at your **own** past research
and letting it hurt. This is that.

## The setup

In mid-2026 I evaluated a TradingView strategy ("r17") for 0DTE SPY trading in a private
research repo. The sweep tried **6 configurations per dataset** — 3 higher-timeframe (HTF)
bias variants × 2 direction modes — across 3 SPY intraday datasets:

- The `LEAKY(r17)` variant reproduced the original Pine script faithfully, **including its
  bug**: the HTF bias used `security()` values that repaint intrabar, i.e. lookahead.
- The `causal` variant fixed the leak (HTF values lagged to what was actually knowable).
- `no-HTF` dropped the filter entirely.

Manual verdict at the time: the strategy's apparent edge came from the lookahead bug —
kill it and the edge dies. Verdict reached the artisanal way: staring at trade lists,
finding the repaint, re-running.

## The audit

Question for the toolkit: *if I had naively picked the best in-sample config from the sweep,
what would DSR/CSCV have said — with no knowledge of the bug?*

Per-config trade R-multiples were bucketed into daily P&L, giving a T×N daily return matrix
per dataset, then run through `overfit.analyze()`. Two of the three datasets had too few
trade days to analyze honestly (12 and 19 days) and were skipped rather than forced. The
third (pseudo 5-min bars, Aug 2024–Nov 2025) had 6 configs × 110 trade days:

```
best in-sample config:            LEAKY(r17)|short      ← the buggy one
best annual Sharpe (in-sample):   +0.87
Deflated Sharpe (DSR):            0.34    (want > 0.95)
PBO (CSCV):                       0.60    (want < 0.50)
P(OOS loss of IS-best):           0.64
IS→OOS degradation slope:         −1.44
min backtest length for N=6:      2.2 years  (sample: ~1.2)
VERDICT: OVERFIT
```

## What this shows

1. **The selection process itself gravitates to the bug.** With no code inspection at all,
   "pick the best in-sample Sharpe" chooses exactly the lookahead variant — leaky signals
   are precisely the kind that look great in-sample.
2. **The formal methods convict it.** DSR 0.34 says the +0.87 Sharpe is unremarkable as the
   best of 6 trials; PBO 0.60 says the in-sample winner ranks *below median* out-of-sample
   more often than not; the −1.44 degradation slope says better IS Sharpe actively predicted
   worse OOS Sharpe.
3. **The sample was too short to trust any selection.** Minimum backtest length for 6 trials
   at that Sharpe is 2.2 years; the data covered ~1.2. Even the honest configs couldn't have
   been validated from this sweep alone.
4. Formal deflation reproduced in seconds what originally took a manual bug-hunt — and it
   would have flagged the result even if the bug had never been found.

## Reproduce

The underlying strategy repo is private research, but the audit script pattern is simple and
generic — build a (days × configs) matrix of daily strategy returns and call:

```python
import overfit
rep = overfit.analyze(daily_returns_matrix, periods_per_year=252)
print(overfit.format_report(rep))
```
