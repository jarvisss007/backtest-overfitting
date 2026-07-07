"""
Reproduce the daily mean-reversion case study (case_studies/daily-meanrev.md).

Builds an 18-variant after-cost daily return matrix for SPY streak strategies
and runs the overfitting workup, plus a walk-forward yearly re-selection.

Needs: pip install yfinance   (public data; everything else is repo deps)
Run:   python case_studies/daily_meanrev_repro.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from overfit import analyze, format_report, sharpe_ratio

COST_PER_SIDE_BPS = 1.0


def load_returns(start="1994-01-01"):
    import yfinance as yf

    px = yf.download("SPY", start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.DataFrame):
        px = px.iloc[:, 0]
    return px.pct_change().dropna()


def build_variants(r):
    """18 positions series; everything at index t uses data <= t only."""
    sign = np.sign(r)
    runs = (sign != sign.shift()).cumsum()
    streak = (sign.groupby(runs).cumcount() + 1) * sign
    vol21 = r.rolling(21).std()
    vol_med = vol21.rolling(252).median()
    regime = pd.Series(np.where(vol21 > vol_med, "highvol", "lowvol"), index=r.index)
    regime[vol_med.isna()] = "na"

    variants = {}
    for k in (1, 2, 3):
        for sig, side in (((streak <= -k).astype(float), "dnL"),
                          (-(streak >= k).astype(float), "upS")):
            for reg in ("all", "highvol", "lowvol"):
                pos = sig if reg == "all" else sig.where(regime == reg, 0.0)
                variants[f"{side}_k{k}_{reg}"] = pos.fillna(0.0)
    return variants


def net_returns(pos, r_next, cost_side_bps=COST_PER_SIDE_BPS):
    turnover = pos.diff().abs().fillna(pos.abs())
    return pos * r_next - turnover * cost_side_bps / 1e4


def walk_forward(net, first_year=1999, lookback_years=5):
    oos = []
    for y in sorted({d.year for d in net.index}):
        if y < first_year:
            continue
        hist = net[(net.index.year >= y - lookback_years) & (net.index.year < y)]
        if len(hist) < 500:
            continue
        srs = hist.apply(lambda c: sharpe_ratio(c.values))
        yr = net.loc[net.index.year == y]
        oos.append(yr[srs.idxmax()] if srs.max() > 0
                   else pd.Series(0.0, index=yr.index))
    return pd.concat(oos).sort_index()


def summarize(x):
    sr = sharpe_ratio(x.values) * np.sqrt(252)
    return f"ann={x.mean() * 252 * 100:+6.2f}%  Sharpe={sr:+5.2f}"


if __name__ == "__main__":
    r = load_returns()
    r_next = r.shift(-1)
    variants = build_variants(r)
    net = pd.DataFrame(
        {n: net_returns(p, r_next) for n, p in variants.items()}
    ).dropna()
    print(f"SPY {r.index[0].date()} -> {r.index[-1].date()}, "
          f"{len(variants)} variants, {COST_PER_SIDE_BPS:.0f} bp/side costs\n")

    rep = analyze(net.values, periods_per_year=252, n_splits=16)
    rep["best_strategy"] = net.columns[rep["best_strategy"]]
    print(format_report(rep))

    oos = walk_forward(net)
    bh = r_next.loc[oos.index]
    print(f"\nWalk-forward adaptive OOS {oos.index[0].year}-{oos.index[-1].year}: "
          f"{summarize(oos)}")
    print(f"Buy-and-hold SPY, same span:              {summarize(bh)}")
