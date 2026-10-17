"""
Tests for case study 5 (case_studies/ml_model_search/run_search.py): a model search on the null world.

Three claims are tested, in the order the case study makes them:
  1. the null world has no edge by construction (no mean, no autocorrelation, no direction
     predictability; mean out-of-sample Sharpe over seeds is zero);
  2. a model search on it still produces a positive in-sample "winner" almost every time;
  3. the toolkit's DSR / PBO flag that winner.
Plus two guards on the engine itself: the walk-forward has no lookahead, and the same engine
DOES find an edge when one is planted (a tool that finds nothing anywhere proves nothing).
Finally, the committed results.json / results.md are checked against each other.

Needs scikit-learn (see requirements.txt); skipped, not failed, where it is missing.
"""
import json
import os
import sys

import numpy as np
import pytest

pytest.importorskip("sklearn")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CS = os.path.join(ROOT, "case_studies", "ml_model_search")
sys.path.insert(0, ROOT)
sys.path.insert(0, CS)
import run_search as rs  # noqa: E402
from overfit import min_backtest_length, sharpe_ratio  # noqa: E402

TINY = dict(rs.FAST, burn_in=100, is_days=160, oos_days=80, refit_every=40)


# ------------------------------------------------------------------ 1. the null world has no edge
def test_null_world_has_no_predictable_structure():
    n = 200_000
    p = rs.simulate_null_world(n, seed=11)
    assert len(p) == n + 1 and np.all(p > 0)
    r = p[1:] / p[:-1] - 1.0
    assert abs(r.mean() / r.std() * np.sqrt(n)) < 3.5                    # zero mean
    for lag in range(1, 6):                                               # no linear predictability
        assert abs(np.corrcoef(r[:-lag], r[lag:])[0, 1]) < 4.0 / np.sqrt(n)
    same_sign = np.mean(np.sign(r[1:]) == np.sign(r[:-1]))               # no directional memory
    assert abs(same_sign - 0.5) < 4 * 0.5 / np.sqrt(n)
    # ...but it is not i.i.d. noise: volatility clusters, as in the real series it stands in for
    assert np.corrcoef(r[:-1] ** 2, r[1:] ** 2)[0, 1] > 0.1


def test_null_world_is_seeded():
    a, b = rs.simulate_null_world(500, 3), rs.simulate_null_world(500, 3)
    c = rs.simulate_null_world(500, 4)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_mean_oos_sharpe_is_zero_over_seeds():
    """Cheap models, many seeds: the out-of-sample Sharpe of a fitted model on the null world
    averages to zero before costs, and costs only push it down."""
    cfg = rs.FAST
    gross, net = [], []
    for seed in range(1000, 1060):
        p = rs.simulate_null_world(cfg["burn_in"] + 1 + cfg["is_days"] + cfg["oos_days"], seed)
        r = p[1:] / p[:-1] - 1.0
        g, n_ = [], []
        for w in rs.WINDOWS:
            a, b = rs.run_candidate(r, "ridge", w, cfg, seed)
            g.append(sharpe_ratio(b[cfg["is_days"]:], rs.PPY))
            n_.append(sharpe_ratio(a[cfg["is_days"]:], rs.PPY))
        gross.append(np.mean(g))
        net.append(np.mean(n_))
    gross, net = np.array(gross), np.array(net)
    se = gross.std(ddof=1) / np.sqrt(len(gross))
    assert abs(gross.mean()) < 3 * se            # statistically zero
    assert abs(gross.mean()) < 0.3               # and small in absolute terms
    assert net.mean() < gross.mean()             # costs can only hurt


# ------------------------------------------------------------------ engine guards
@pytest.mark.parametrize("model,window", [("ridge", 5), ("gbm", 10), ("mlp", 5), ("rf", 20)])
def test_walk_forward_has_no_lookahead(model, window):
    cfg = TINY
    n = cfg["burn_in"] + 1 + cfg["is_days"] + cfg["oos_days"]
    p = rs.simulate_null_world(n, 21)
    r = p[1:] / p[:-1] - 1.0
    m = n - 70
    r2 = r.copy()
    r2[m:] = np.random.default_rng(5).normal(0, 0.03, size=n - m)      # rewrite the future
    a, _ = rs.run_candidate(r, model, window, cfg, 7)
    b, _ = rs.run_candidate(r2, model, window, cfg, 7)
    k = m - cfg["burn_in"] - 1                  # stream days before the rewrite begins
    assert k > 100
    assert np.array_equal(a[:k], b[:k])         # nothing before day m moved
    assert not np.array_equal(a[k:], b[k:])     # and the change is visible afterwards


def test_engine_finds_a_planted_edge():
    cfg = rs.FAST
    n = cfg["burn_in"] + 1 + cfg["is_days"] + cfg["oos_days"]
    rng = np.random.default_rng(8)
    eps = rng.normal(0, 0.01, size=n)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = 0.2 * r[t - 1] + eps[t]            # genuine one-day momentum
    net, gross = rs.run_candidate(r, "ridge", 5, cfg, 1)
    assert sharpe_ratio(gross, rs.PPY) > 1.5
    assert sharpe_ratio(net, rs.PPY) > 1.0        # survives the 1 bp/side costs


def test_summary_structure_and_selection_rule():
    res = rs.run_search(5, TINY, n_jobs=1)
    L = TINY["is_days"] + TINY["oos_days"]
    assert res["net"].shape == (L, 20) and res["gross"].shape == (L, 20)
    s = rs.summarize(res, n_splits=4)
    sel = s["selection"]
    is_sr = [c["is_sharpe"] for c in s["surface"]]
    assert len(s["surface"]) == 20
    assert {(c["model"], c["window"]) for c in s["surface"]} == set(rs.candidates())
    assert sel["winner_index"] == int(np.argmax(is_sr))                 # picked on the backtest only
    assert sel["winner_is_sharpe"] == pytest.approx(max(is_sr))
    assert s["toolkit"]["n_trials"] == 20


# ------------------------------------------------------------------ 2 + 3. the search and the deflation
@pytest.fixture(scope="module")
def replication():
    recs = rs.replicate(12, rs.FAST, base_seed=424242, n_jobs=min(4, os.cpu_count() or 1), n_splits=8)
    return recs, rs.summarize_replication(recs, rs.FAST)


def test_search_picks_a_positive_in_sample_winner(replication):
    _, s = replication
    assert s["share_positive_is_winner"] >= 0.85          # best of 20 noisy Sharpes is almost always > 0
    assert s["mean_winner_is_sharpe"] > 0.4               # and looks like a real signal on average
    # ...yet nothing carries over: out-of-sample the winner is zero (gross) within noise, and the
    # drop from the backtest is large
    assert abs(s["mean_winner_oos_sharpe_gross"]) < 3 * s["se_winner_oos_sharpe_gross"]
    assert s["mean_winner_is_sharpe"] - s["mean_winner_oos_sharpe"] > 0.3


def test_toolkit_flags_the_winner(replication):
    recs, s = replication
    assert s["share_dsr_below_095"] >= 0.85               # DSR: not significant after deflation
    assert s["share_flagged_by_verdict"] >= 0.85          # verdict is never "survives" (or very rarely)
    assert s["mean_pbo"] >= 0.35                          # PBO ~ 0.5 on pure noise: a coin flip
    # no seed should clear BOTH gates -- that would be a false pass
    assert not any(r["dsr"] >= 0.95 and r["pbo"] < 0.5 for r in recs)


# ------------------------------------------------------------------ the committed results
def _load():
    with open(os.path.join(CS, "results.json")) as f:
        return json.load(f)


def test_committed_results_are_consistent():
    out = _load()
    surf, sel, tk = out["surface"], out["selection"], out["toolkit"]
    assert len(surf) == 20
    assert {(c["model"], c["window"]) for c in surf} == set(rs.candidates())
    top = max(surf, key=lambda c: c["is_sharpe"])
    assert sel["winner"] == f"{top['model']}|w{top['window']}"          # picked on the backtest only
    assert sel["winner_is_sharpe"] == pytest.approx(top["is_sharpe"], abs=1e-3)
    assert sel["winner_is_sharpe"] > 0                                  # a positive "winner" emerged
    assert tk["best_sharpe_annual"] == pytest.approx(sel["winner_is_sharpe"], abs=1e-3)
    assert tk["n_trials"] == 20
    assert tk["min_backtest_length_years"] == pytest.approx(
        min_backtest_length(20, tk["best_sharpe_annual"]), abs=1e-2)
    assert tk["dsr"] < 0.95                                             # DSR flags the winner
    assert out["meta"]["seed"] == rs.SEED


def test_results_md_quotes_the_json():
    out = _load()
    sel, tk = out["selection"], out["toolkit"]
    with open(os.path.join(CS, "results.md")) as f:
        md = f.read()
    rep = out["replication"]["summary"]
    n = rep["n_seeds"]
    needles = [f"{sel['winner_is_sharpe']:+.2f}", f"{sel['winner_oos_sharpe']:+.2f}",
               f"{tk['dsr']:.2f}", f"{tk['pbo']:.2f}", f"{tk['min_backtest_length_years']:.1f}",
               f"{sel['spearman_is_vs_oos']:+.2f}",
               f"{rep['mean_winner_is_sharpe']:+.2f}", f"{rep['mean_winner_oos_sharpe']:+.2f}",
               f"{rep['mean_pbo']:.2f}", f"{rep['median_dsr']:.2f}",
               f"{round(rep['share_dsr_below_095'] * n)} / {n}",
               f"{round(rep['share_positive_is_winner'] * n)} / {n}",
               f"{round(rep['share_pbo_at_least_half'] * n)} / {n}"]
    for needle in needles:
        assert needle in md, needle
