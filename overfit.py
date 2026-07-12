"""
backtest-overfitting — tools for detecting when a backtest is fooling you.

Trying many strategy configurations and reporting the best one's Sharpe is the most common
way to publish a spurious "edge". These tools quantify how much of an apparent edge is likely
an artifact of selection / multiple testing.

Implements:
  probabilistic_sharpe_ratio  — is a Sharpe significantly above a benchmark, given skew/kurtosis?
  expected_max_sharpe         — the Sharpe you'd expect from the *best of N* random trials
  deflated_sharpe_ratio       — PSR against that inflated benchmark (Bailey & López de Prado)
  min_track_record_length     — how long a track record must be to trust a Sharpe
  min_backtest_length         — max in-sample Sharpe you'd get by luck from N trials → years needed
  pbo_cscv                    — Probability of Backtest Overfitting via combinatorially
                                symmetric cross-validation (Bailey, Borwein, López de Prado, Zhu)
  analyze                     — tie it together for a (T × N) matrix of strategy returns

References:
  Bailey & López de Prado (2014), "The Deflated Sharpe Ratio", J. Portfolio Management.
  Bailey, Borwein, López de Prado & Zhu (2015), "The Probability of Backtest Overfitting",
    J. Computational Finance.
  Bailey, Borwein, López de Prado & Zhu (2014), "Pseudo-Mathematics and Financial
    Charlatanism", Notices of the AMS.

Convention: Sharpe ratios passed to the *statistical* functions are PER-PERIOD (not annualized)
unless noted; kurtosis is NON-EXCESS (normal = 3).
"""
from itertools import combinations
import numpy as np
from scipy.stats import norm, skew as _skew, kurtosis as _kurtosis

EULER = 0.5772156649015329  # Euler–Mascheroni constant


# ------------------------------------------------------------------ basics
def sharpe_ratio(returns, periods_per_year=1):
    """Sharpe of a return series. periods_per_year>1 annualizes it."""
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    return (r.mean() / sd) * np.sqrt(periods_per_year)


def _moments(returns):
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=1)
    sr = 0.0 if sd == 0 else r.mean() / sd
    return sr, _skew(r), _kurtosis(r, fisher=False), len(r)


# ------------------------------------------------------------------ PSR
def probabilistic_sharpe_ratio(sr, n_obs, skew=0.0, kurtosis=3.0, sr_benchmark=0.0):
    """
    P(true Sharpe > sr_benchmark), given an observed per-period Sharpe `sr`, sample length
    `n_obs`, and the returns' skew/kurtosis. Returns a probability in [0, 1].
    """
    denom = np.sqrt(1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr ** 2)
    if denom == 0 or n_obs < 2:
        return float("nan")
    z = (sr - sr_benchmark) * np.sqrt(n_obs - 1.0) / denom
    return float(norm.cdf(z))


def min_track_record_length(sr, skew=0.0, kurtosis=3.0, sr_benchmark=0.0, confidence=0.95):
    """Minimum # of observations for PSR(sr_benchmark) to reach `confidence`. NaN if sr<=benchmark."""
    if sr <= sr_benchmark:
        return float("nan")
    z = norm.ppf(confidence)
    adj = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr ** 2
    return 1.0 + adj * (z / (sr - sr_benchmark)) ** 2


# ------------------------------------------------------------------ deflation / multiple testing
def expected_max_sharpe(sr_variance, n_trials, mean_sr=0.0):
    """
    Expected maximum per-period Sharpe from the *best of `n_trials`* strategies whose true
    Sharpes are noise with variance `sr_variance`. This is the benchmark a real edge must beat.
    """
    if n_trials < 2:
        return mean_sr
    s = np.sqrt(sr_variance)
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return mean_sr + s * ((1.0 - EULER) * z1 + EULER * z2)


def deflated_sharpe_ratio(sr, n_obs, sr_variance, n_trials, skew=0.0, kurtosis=3.0, mean_sr=0.0):
    """
    Deflated Sharpe Ratio: PSR evaluated against the inflated benchmark you'd expect from
    picking the best of `n_trials`. Returns (dsr, benchmark_sr). dsr<0.95 ⇒ the edge is
    plausibly just the luckiest of many tries.
    """
    sr0 = expected_max_sharpe(sr_variance, n_trials, mean_sr)
    dsr = probabilistic_sharpe_ratio(sr, n_obs, skew, kurtosis, sr_benchmark=sr0)
    return dsr, sr0


def min_backtest_length(n_trials, target_sharpe_annual):
    """
    Minimum backtest length (years) so the expected best-of-`n_trials` in-sample annualized
    Sharpe (true edge = 0) does not spuriously reach `target_sharpe_annual`.
    Rule of thumb from "Pseudo-Mathematics and Financial Charlatanism".
    """
    if target_sharpe_annual <= 0:
        return float("inf")
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    e_max = (1.0 - EULER) * z1 + EULER * z2
    return (e_max / target_sharpe_annual) ** 2


# ------------------------------------------------------------------ PBO via CSCV
def pbo_cscv(returns, n_splits=16):
    """
    Probability of Backtest Overfitting via combinatorially symmetric cross-validation.

    `returns`: (T observations × N strategies) matrix. Splits time into `n_splits` chunks,
    and over every way of choosing half as in-sample (IS) / half as out-of-sample (OOS):
      - pick the strategy with the best IS Sharpe,
      - see where it ranks OOS.
    If the IS-best routinely lands in the bottom half OOS, the selection is overfit.

    Returns a dict:
      pbo            — P(OOS rank of the IS-best is below median). >0.5 is bad.
      prob_oos_loss  — fraction of splits where the IS-best has negative OOS Sharpe.
      degradation    — OLS slope of OOS Sharpe on IS Sharpe (negative ⇒ overfitting).
      logits, is_sr, oos_sr — per-combination arrays (for plotting).
    """
    M = np.asarray(returns, dtype=float)
    T, N = M.shape
    if N < 2:
        raise ValueError("need >= 2 strategies")
    rows = (T // n_splits) * n_splits
    M = M[:rows]
    chunks = np.array_split(np.arange(rows), n_splits)

    # per-chunk sufficient statistics per strategy (for fast Sharpe over any chunk subset)
    c_sum = np.array([M[idx].sum(axis=0) for idx in chunks])          # (S, N)
    c_sq = np.array([(M[idx] ** 2).sum(axis=0) for idx in chunks])    # (S, N)
    c_n = np.array([len(idx) for idx in chunks])                       # (S,)

    def sharpe(chunk_ids):
        n = c_n[list(chunk_ids)].sum()
        s = c_sum[list(chunk_ids)].sum(axis=0)
        sq = c_sq[list(chunk_ids)].sum(axis=0)
        mean = s / n
        var = sq / n - mean ** 2
        sd = np.sqrt(np.clip(var, 0, None))
        out = np.zeros(N)
        nz = sd > 0
        out[nz] = mean[nz] / sd[nz]
        return out

    all_ids = set(range(n_splits))
    logits, is_best, oos_best = [], [], []
    for is_ids in combinations(range(n_splits), n_splits // 2):
        oos_ids = tuple(all_ids - set(is_ids))
        is_sr = sharpe(is_ids)
        oos_sr = sharpe(oos_ids)
        n_star = int(np.argmax(is_sr))
        # relative OOS rank of the IS-best in (0,1)
        rank = int((oos_sr <= oos_sr[n_star]).sum())      # 1..N
        omega = rank / (N + 1.0)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        logits.append(np.log(omega / (1.0 - omega)))
        is_best.append(is_sr[n_star])
        oos_best.append(oos_sr[n_star])

    logits = np.array(logits)
    is_best = np.array(is_best)
    oos_best = np.array(oos_best)
    slope = np.polyfit(is_best, oos_best, 1)[0] if len(is_best) > 1 else float("nan")
    return {
        "pbo": float(np.mean(logits <= 0)),
        "prob_oos_loss": float(np.mean(oos_best < 0)),
        "degradation": float(slope),
        "n_combinations": len(logits),
        "logits": logits,
        "is_sr": is_best,
        "oos_sr": oos_best,
    }


# ------------------------------------------------------------------ one-call report
def analyze(returns, periods_per_year=252, n_splits=16):
    """
    Full overfitting workup on a (T × N) matrix of strategy returns. Selects the best strategy
    by full-sample Sharpe, then reports whether that selection survives deflation & CSCV.
    """
    M = np.asarray(returns, dtype=float)
    T, N = M.shape
    full_sr = np.array([sharpe_ratio(M[:, j]) for j in range(N)])   # per-period
    best = int(np.argmax(full_sr))
    sr_p, sk, ku, n = _moments(M[:, best])
    sr_variance = float(np.var(full_sr, ddof=1))

    psr = probabilistic_sharpe_ratio(sr_p, n, sk, ku, sr_benchmark=0.0)
    dsr, sr0 = deflated_sharpe_ratio(sr_p, n, sr_variance, N, sk, ku)
    ann = np.sqrt(periods_per_year)
    minbtl = min_backtest_length(N, sr_p * ann)
    cscv = pbo_cscv(M, n_splits=n_splits) if N >= 2 else None

    return {
        "n_strategies": N, "n_obs": T, "best_strategy": best,
        "best_sharpe_annual": float(sr_p * ann),
        "psr": psr, "dsr": dsr,
        "deflated_benchmark_sharpe_annual": float(sr0 * ann),
        "min_backtest_length_years": float(minbtl),
        "pbo": cscv["pbo"] if cscv else float("nan"),
        "prob_oos_loss": cscv["prob_oos_loss"] if cscv else float("nan"),
        "degradation": cscv["degradation"] if cscv else float("nan"),
        "verdict": _verdict(dsr, cscv["pbo"] if cscv else float("nan")),
    }


def monte_carlo_drawdown(returns, n_sims=5000, seed=0):
    """The specific exercise from the Fluency Project's Complete Guide to Day Trading
    (Heitkoetter): don't trust the one historical drawdown you happened to observe —
    shuffle the trade sequence thousands of times (same trades, different order) and
    look at the DISTRIBUTION of max drawdown. A backtest with a mild-looking 8% max
    drawdown can still have a 25% worst-case drawdown lurking in a different, equally
    likely ordering of the same trades. This complements deflated_sharpe_ratio/pbo_cscv
    (which ask "is the edge real?") by asking a different question: "if it's real, how
    bad could living through it actually get?"

    Returns dict with the actual historical max drawdown plus the simulated
    distribution's percentiles (5/25/50/75/95), all as fractions (0.08 = 8%).
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 30:
        return {"error": "fewer than 30 returns — not enough trades to shuffle meaningfully"}

    def max_dd(seq):
        equity = np.cumprod(1.0 + seq)
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        return float(-dd.min())

    actual_dd = max_dd(r)
    rng = np.random.default_rng(seed)
    sims = np.empty(n_sims)
    for i in range(n_sims):
        sims[i] = max_dd(rng.permutation(r))

    pct = lambda q: float(np.percentile(sims, q))
    return {
        "n_trades": len(r), "n_sims": n_sims,
        "actual_order_max_dd": actual_dd,
        "sim_p5": pct(5), "sim_p25": pct(25), "sim_p50": pct(50),
        "sim_p75": pct(75), "sim_p95": pct(95),
        "actual_was_lucky_ordering": bool(actual_dd < pct(25)),
    }


def _verdict(dsr, pbo):
    if np.isnan(dsr):
        return "inconclusive"
    if dsr >= 0.95 and pbo < 0.5:
        return "survives: significant after deflation and low overfitting probability"
    if dsr < 0.95 and pbo >= 0.5:
        return "OVERFIT: not significant after deflation AND high overfitting probability"
    return "suspect: fails one of {deflated significance, low PBO} — treat as unproven"


def format_report(rep):
    """Pretty one-screen summary of analyze()."""
    L = [
        "  " + "=" * 60,
        f"  Backtest overfitting report — {rep['n_strategies']} strategies, "
        f"{rep['n_obs']} obs",
        "  " + "-" * 60,
        f"  Best strategy (#{rep['best_strategy']}) Sharpe (ann.):   "
        f"{rep['best_sharpe_annual']:.2f}",
        f"  Probabilistic Sharpe (vs 0):                {rep['psr']:.3f}",
        f"  Deflated Sharpe (vs best-of-N benchmark):   {rep['dsr']:.3f}   "
        f"(benchmark {rep['deflated_benchmark_sharpe_annual']:.2f} ann.)",
        f"  Prob. of Backtest Overfitting (PBO):        {rep['pbo']:.3f}   "
        f"(>0.5 is bad)",
        f"  Prob. OOS loss when selecting IS-best:      {rep['prob_oos_loss']:.3f}",
        f"  IS→OOS performance degradation (slope):     {rep['degradation']:.3f}   "
        f"(negative ⇒ overfitting)",
        f"  Min backtest length for this many trials:   "
        f"{rep['min_backtest_length_years']:.1f} years",
        "  " + "-" * 60,
        f"  VERDICT: {rep['verdict']}",
        "  " + "=" * 60,
    ]
    return "\n".join(L)
