# Case study: the edge in the untradeable gap

The first case study caught a **lookahead bug in code**. The second caught a **real anomaly
too weak to trade**. This one catches the third and subtlest failure mode: a backtest whose
entire edge lives in an **execution assumption** — every line of code correct, every number
real, and still nothing you can trade.

## The setup

I run a live pipeline ([insider-radar](https://github.com/jarvisss007/insider-radar)) that
collects SEC Form 4 filings and keeps insiders' **open-market purchases** — an executive
spending their own cash on their own stock, the classic "smart money" signal with decades of
academic literature behind it.

Event study on ~200 hygiene-filtered events (sub-$1 and split-glitch tickers excluded),
market-adjusted vs SPY, **entering at the filing-day close**:

```
+1d  +1.04%  t=2.74      win 60.5%
+3d  +1.74%  t=3.47
+5d  +1.84%  t=3.05
+10d +1.84%  t=2.17
+20d −0.24%  t=−0.23     (edge fully decayed)
```

Calendar-time portfolio, 40 bps round-trip costs: **net Sharpe +3.1** on the 1-day hold.
This toolkit's verdict at that point: DSR 0.81, PBO 0.37 → *"suspect — treat as unproven"* —
promising, but not yet distinguishable from a lucky 5-variant selection on a short sample.

## The kill

One realism question destroyed it. Form 4 filings mostly hit EDGAR **after the market
closes** — so "buy at the filing-day close" is an entry you usually cannot get. The honest
entry is the **next trading day's open**.

Re-run everything with strict next-open entry, same events, same hygiene, same costs:

```
event study:  NO significant edge at any horizon (all |t| < 2)
portfolio:    every hold-variant net-NEGATIVE (best Sharpe −0.15)
toolkit:      DSR 0.20, PBO 0.60  →  OVERFIT / no edge
```

The entire effect lives in the **overnight gap between the after-hours filing and the next
open**. The market reprices the insider's purchase before any follower can act. The original
backtest wasn't wrong about the market — it was wrong about *you*: it silently assumed you
could trade at a price that had already expired.

## Large-sample confirmation (and one more lesson)

A deeper backfill (deterministic 60-filings/day sample, Jul 2025 – Jul 2026) grew the set to
**675 clean events / 624 mature across 378 tickers** — a full year. At strict entry the
means look significant again… and the distribution tells you why that's a trap:

```
+1d   mean +0.70%   t=3.76     median +0.01%   win 50.5%
+5d   mean +0.89%   t=2.66     median −0.01%   win 49.7%
+20d  mean +1.03%   t=1.38     median −0.70%   win 45.7%

portfolio (254 days × 5 holds, 40 bps):  best net Sharpe +0.54
toolkit:  DSR 0.41 · PBO 0.914  →  OVERFIT / no tradeable edge
```

A t-stat of 3.8 with a **median of zero and a coin-flip win rate**: the "signal" is a
handful of extreme right-tail winners, not a repeatable tendency. The calendar-time
portfolio confirms it — hold-period selection is wildly unstable (PBO 0.91, the in-sample
best ranks below median out-of-sample 9 times in 10). Lesson four, free of charge:
**on skewed event distributions, a significant mean is not a strategy — read the median,
the win rate, and the gate.**

## What this teaches

1. **Entry convention is a hypothesis, not a detail.** One line — `close[t]` vs
   `open[t+1]` — flipped a t=3.5 signal into a null. Every event-driven backtest should be
   run at the *worst plausible* entry before being believed.
2. **Announcement effects are real and unreachable.** The event study is honest — prices do
   jump on insider buys. "The signal exists" and "you can capture it" are different claims;
   only the second pays.
3. **The gate can't catch what the simulation assumes away.** DSR/PBO correctly flagged the
   close-entry version as *unproven* rather than proven — but only the execution-realism
   pass produced the definitive kill. Statistical deflation and simulation realism are
   complementary, not substitutes.

## Reproduce

The pipeline (collector, event study, validator) is public in
[insider-radar](https://github.com/jarvisss007/insider-radar):

```
python collector_edgar.py            # live Form 4 feed
python backfill_edgar.py --start 2026-03-02 --end 2026-07-02 --per-day 60
python event_study.py                # strict next-open entry (the honest default)
python research/validate_signal.py   # calendar-time portfolio through this toolkit
```
