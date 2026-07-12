"""Property + behaviour tests. Run: pytest -q"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from overfit import (probabilistic_sharpe_ratio, expected_max_sharpe,
                     deflated_sharpe_ratio, min_track_record_length,
                     min_backtest_length, pbo_cscv, analyze, monte_carlo_drawdown)


def test_psr_bounds_and_monotonic():
    p_lo = probabilistic_sharpe_ratio(0.05, 500)
    p_hi = probabilistic_sharpe_ratio(0.20, 500)
    assert 0.0 <= p_lo <= 1.0 and 0.0 <= p_hi <= 1.0
    assert p_hi > p_lo                      # higher Sharpe => higher PSR
    # longer sample => more confident for the same positive Sharpe
    assert probabilistic_sharpe_ratio(0.1, 2000) > probabilistic_sharpe_ratio(0.1, 200)


def test_expected_max_sharpe_increases_with_trials():
    a = expected_max_sharpe(0.01, 10)
    b = expected_max_sharpe(0.01, 100)
    c = expected_max_sharpe(0.01, 1000)
    assert a < b < c                        # more trials => higher lucky max
    assert expected_max_sharpe(0.01, 1) == 0.0


def test_deflation_lowers_significance():
    # same Sharpe is less impressive if it's the best of many trials
    dsr1, sr0_1 = deflated_sharpe_ratio(0.1, 500, sr_variance=0.01, n_trials=2)
    dsr50, sr0_50 = deflated_sharpe_ratio(0.1, 500, sr_variance=0.01, n_trials=50)
    assert sr0_50 > sr0_1
    assert dsr50 < dsr1


def test_min_backtest_length_grows_with_trials():
    assert min_backtest_length(1000, 1.0) > min_backtest_length(10, 1.0)
    assert min_backtest_length(100, 2.0) < min_backtest_length(100, 1.0)  # higher bar, but harder Sharpe needs less time


def test_mintrl_nan_when_below_benchmark():
    assert np.isnan(min_track_record_length(0.05, sr_benchmark=0.10))
    assert min_track_record_length(0.20, sr_benchmark=0.0) > 0


def test_pbo_high_on_pure_noise():
    rng = np.random.default_rng(0)
    M = rng.normal(0, 0.01, size=(800, 100))        # no edge anywhere
    res = pbo_cscv(M, n_splits=14)
    assert 0.0 <= res["pbo"] <= 1.0
    assert res["pbo"] > 0.35                         # selection on noise => substantial overfitting
    assert res["degradation"] < 0.5                  # IS success does not carry to OOS


def test_analyze_flags_noise_and_passes_real_edge():
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 0.01, size=(1000, 200))
    rep_noise = analyze(noise)
    assert rep_noise["dsr"] < 0.95                    # cherry-picked noise not significant
    assert "OVERFIT" in rep_noise["verdict"] or "suspect" in rep_noise["verdict"]

    real = rng.normal(0, 0.01, size=(1000, 200))
    real[:, 0] += 0.01 * 1.5 / np.sqrt(252)          # a genuine ~1.5 Sharpe edge
    rep_real = analyze(real)
    assert rep_real["best_strategy"] == 0             # it should be selected
    assert rep_real["dsr"] > rep_noise["dsr"]         # and be more credible than pure noise


def test_monte_carlo_drawdown_shape():
    rng = np.random.default_rng(1)
    r = rng.normal(0.0005, 0.01, size=500)
    res = monte_carlo_drawdown(r, n_sims=1000, seed=1)
    assert res["n_trades"] == 500
    assert 0.0 <= res["sim_p5"] <= res["sim_p25"] <= res["sim_p50"] <= res["sim_p75"] <= res["sim_p95"]
    assert res["actual_order_max_dd"] >= 0.0


def test_monte_carlo_drawdown_flags_unlucky_historical_ordering():
    # same 500 trades every time; but the ACTUAL historical order dumps every loss
    # consecutively at the start — the worst possible ordering — while a typical
    # shuffle spreads gains and losses out and drawdown stays much milder.
    rng = np.random.default_rng(2)
    wins = np.full(400, 0.01)
    losses = np.full(100, -0.03)
    worst_order = np.concatenate([losses, wins])       # all pain up front
    res = monte_carlo_drawdown(worst_order, n_sims=2000, seed=2)
    assert res["actual_order_max_dd"] >= res["sim_p95"]          # this ordering is in the bad tail
    assert res["actual_was_lucky_ordering"] is False

    # the worst-case ordering's drawdown should be materially worse than the AVERAGE
    # random shuffle of the same trades (a single random draw can itself land in
    # either tail sometimes, so compare against the mean of several, not one draw)
    typical_dds = [monte_carlo_drawdown(rng.permutation(worst_order), n_sims=1)["actual_order_max_dd"]
                   for _ in range(30)]
    assert res["actual_order_max_dd"] > np.mean(typical_dds)
