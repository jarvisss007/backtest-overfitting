# Case study: passing every gate still isn't the whole story

The [first case study](zero-dte-audit.md) convicted a fake edge. The
[second](daily-meanrev.md) showed a real-but-untradeable anomaly. This one is about
the best-scoring hypothesis `~/strategy-lab/discover.py` has ever produced — and what
DSR and PBO *don't* tell you about it.

## The setup

`turn-of-month t5` (last 5 + first 5 trading days of the month, cross-sectional basket,
15 years of daily bars): DSR **1.0**, PBO **0.01**, walk-forward **5/5** — the cleanest
statistical profile of anything in the knowledge base. **Verdict: beats zero, not the
market.** Its own verdict is "loses to buy-and-hold": net Sharpe 1.05, below the
basket's own buy-and-hold Sharpe. This repo does not hold the data or script behind
that buy-and-hold figure, so no number is given here (see `~/strategy-lab/discover.py`,
which produced the verdict).

Why the gates pass anyway: DSR and PBO check whether the Sharpe beats zero (DSR also
corrects for the number of configs tried) and whether the winner holds up out of sample.
Neither compares the strategy to buy-and-hold. A long-only equity basket collects the
equity premium, so almost any such basket clears a test against zero.

DSR and PBO also say nothing about what living through it would actually feel like.

That ride question is Heitkoetter's (*A Complete Guide to Day Trading*, Fluency
Project Book 6): shuffle the same trades into thousands of different orderings and
look at the distribution of max drawdown, because the one historical ordering you
observed is a single draw from a much wider range of equally-likely paths.

## What `monte_carlo_drawdown` found

5,000 reshufflings of the strategy's 3,750 net daily returns:

| Percentile | Max drawdown |
|---|---|
| p5 (lucky) | 19.2% |
| p25 | 22.6% |
| p50 (typical) | 26.0% |
| p75 | 30.0% |
| p95 (unlucky) | 37.6% |
| **Actual historical ordering** | **28.2%** — squarely typical, not a lucky draw |

## The lesson

A DSR of 1.0 and a PBO of 0.01 say this strategy beats zero, not the market: it still
loses to buy-and-hold, because the gates never ask that question. They also don't say
the *ride* is comfortable — this strategy's honest range of plausible max drawdown spans
19%–38% depending on nothing but the order the same trades happened to arrive in.
The actual historical result (28.2%) sits right at the median: not lucky, not unlucky,
just what a ~26% typical drawdown looks like when it happens to you specifically.

**Takeaway:** run `monte_carlo_drawdown` on anything that survives DSR/PBO before
believing the historical drawdown number is *the* drawdown number. It's one sample
from a distribution, and the distribution is usually wider than the single line on
the equity curve suggests.
