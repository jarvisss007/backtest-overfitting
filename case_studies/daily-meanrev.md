# Case study: a REAL anomaly that still isn't tradeable

The [first case study](zero-dte-audit.md) showed the toolkit convicting a fake edge (a
lookahead bug). This one shows the opposite — and arguably more useful — failure mode:
a pattern that is **statistically genuine** and still fails as a strategy. The two
verdicts together are the point: DSR and PBO measure *different things*, and you need
both.

## The setup

Classic short-term mean reversion in SPY daily returns: "after a few straight down
days, the next day bounces." Tested on 32 years of daily bars (1994–2026, ~8,200 obs),
constructed honestly:

- signals at close *t* use only data ≤ *t*; payoff is day *t+1*'s return
- volatility regimes from trailing windows only (no full-sample fits)
- 1 bp/side transaction costs (SPY-realistic), sensitivity at 2 bp/side
- **all 18 variants counted as trials**: streak length k ∈ {1,2,3} × side
  {long-after-down, short-after-up} × regime {all, high-vol, low-vol}
- a walk-forward meta-strategy: each year, trade whichever variant had the best
  trailing 5-year after-cost Sharpe (flat if none was positive)

The raw effect is strong: after ≥3 consecutive down days, next-day SPY averaged
+27 bps (t = 4.3) vs +5 bps unconditionally. Every down-streak-long variant is
profitable after costs; every up-streak-short variant loses.

## The audit

```
best in-sample variant:           long after 3 down days, low-vol regime
best annual Sharpe (in-sample):   +0.80  (after costs)
Probabilistic Sharpe (vs 0):      1.000
Deflated Sharpe (DSR):            0.59    (want > 0.95)
PBO (CSCV):                       0.002   (want < 0.50)
P(OOS loss of IS-best):           0.00
IS→OOS degradation slope:         −0.57
VERDICT: suspect — fails deflated significance despite near-zero PBO
```

And the economics, walk-forward out-of-sample 1999–2026:

```
adaptive meta-strategy:   +2.6%/yr   Sharpe 0.26   maxDD −35%
buy-and-hold SPY:        +10.2%/yr   Sharpe 0.53   maxDD −55%
```

## What this shows

1. **PBO ≈ 0 with DSR < 0.95 is a coherent verdict, not a contradiction.** PBO asks
   "does the in-sample winner keep its *rank* out-of-sample?" — yes, systematically;
   the anomaly is real. DSR asks "is the winner's Sharpe distinguishable from the best
   of 18 lucky coin-flippers?" — no. A real but *weak* edge produces exactly this
   signature. (The 0DTE case was the mirror image: fake edge, PBO 0.60.)
2. **Statistical existence ≠ economic exploitability.** The effect pays ~15–27 bps per
   event, fires rarely (4–20% of days), and clusters in crashes. On one signal and one
   instrument it cannot beat simply holding the index — Sharpe 0.26 vs 0.53.
3. **Adaptation is a parameter that can itself overfit.** The yearly re-selection
   ("trade what worked lately") *degraded* performance — the −0.57 IS→OOS slope showed
   up in practice when the selector chased 2008's winners into short variants for
   2009 and got run over by the rebound.
4. This is presumably why the anomaly persists in public view: the players who could
   harvest 15 bps at scale need hundreds of such signals and near-zero costs; for
   everyone else it's not worth the drawdowns.

## Reproduce

Fully reproducible from public data — `pip install yfinance`, then:

```
python case_studies/daily_meanrev_repro.py
```

builds the 18-variant after-cost return matrix and runs `overfit.analyze()` on it.
