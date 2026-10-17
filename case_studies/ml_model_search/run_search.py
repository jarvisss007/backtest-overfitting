"""
Case study 5: an ML model search on the null world.

Builds a simulated price series in which NOTHING is predictable (zero conditional mean,
volatility clustering kept), runs a researcher's model search on it -- 5 model families x 4
lookback windows = 20 candidates, each fitted walk-forward and traded as sign(prediction) --
selects the "best model" on the backtest, then looks at what happens next.  The whole search
surface (every candidate's in-sample and out-of-sample Sharpe) is reported, and the toolkit's
own DSR / PBO / MinBTL (overfit.py) are applied to the search.

Every "winner" in this world is false by construction: the true Sharpe of every candidate is
zero before costs, so any positive Sharpe is luck.

Pre-registration: the design is in DESIGN.md (same folder).  This script is the reduced
null-world slice of it; the reductions are logged under "Amendments" there.

Run:   python case_studies/ml_model_search/run_search.py
Needs: numpy, scipy, scikit-learn, matplotlib  (joblib and threadpoolctl come with scikit-learn)
Time:  a few minutes on a laptop (the 20-candidate search, then a seed-replication check).
Out:   results.json, results.png in this folder.  (results.md is the written-up reading.)
"""
import argparse
import json
import os
import sys
import time
import warnings
from itertools import product

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from overfit import analyze, format_report, sharpe_ratio  # noqa: E402

SEED = 20261017                 # fixed before the first run; never changed after seeing a result
PPY = 252
COST_PER_SIDE_BPS = 1.0         # 1 bp per unit of position change, so a long->short flip costs 2 bp
MODELS = ("ridge", "lasso", "gbm", "rf", "mlp")
WINDOWS = (5, 10, 20, 60)       # lookback: the model sees the last `w` daily returns

# The null world: GARCH(1,1) volatility with symmetric Student-t innovations and ZERO mean.
# E[r_t | past] = 0 exactly, and sign(r_t) is independent of the past, so no function of
# the past can predict the direction of the next return.
WORLD = dict(daily_vol=0.01, alpha=0.08, beta=0.90, df=6)

# Two configurations of the same experiment.  FULL is the headline run.  FAST is the same
# 20-candidate search on a shorter series with lighter models and quarterly refits; it is what
# the seed-replication and the test-suite use so that many seeds fit in minutes.
FULL = dict(name="full", burn_in=504, is_days=1260, oos_days=756, refit_every=21,
            gbm=dict(n_estimators=50, learning_rate=0.05, max_depth=2, subsample=0.7, max_features=0.33),
            rf=dict(n_estimators=40, min_samples_leaf=25, max_features=0.33, max_samples=0.5),
            mlp=dict(hidden_layer_sizes=(16,), alpha=1e-2, max_iter=100))
FAST = dict(name="fast", burn_in=252, is_days=756, oos_days=504, refit_every=63,
            gbm=dict(n_estimators=20, learning_rate=0.05, max_depth=2, subsample=0.7, max_features=0.33),
            rf=dict(n_estimators=20, min_samples_leaf=25, max_features=0.33, max_samples=0.5, max_depth=6),
            mlp=dict(hidden_layer_sizes=(8,), alpha=1e-2, max_iter=40))


# ------------------------------------------------------------------ the null world
def simulate_null_world(n_returns, seed, daily_vol=None, alpha=None, beta=None, df=None):
    """A seeded price series with no predictable structure.  Returns n_returns + 1 prices."""
    p = dict(WORLD)
    p.update({k: v for k, v in dict(daily_vol=daily_vol, alpha=alpha, beta=beta, df=df).items()
              if v is not None})
    rng = np.random.default_rng(seed)
    n_burn = 500
    n = n_returns + n_burn
    z = rng.standard_t(p["df"], size=n) / np.sqrt(p["df"] / (p["df"] - 2.0))   # unit variance
    omega = p["daily_vol"] ** 2 * (1.0 - p["alpha"] - p["beta"])
    s2 = p["daily_vol"] ** 2
    r = np.empty(n)
    for t in range(n):
        r[t] = np.sqrt(s2) * z[t]
        s2 = omega + p["alpha"] * r[t] ** 2 + p["beta"] * s2
    return 100.0 * np.concatenate([[1.0], np.cumprod(1.0 + r[n_burn:])])


# ------------------------------------------------------------------ one candidate
def _make_model(name, cfg, random_state):
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.neural_network import MLPRegressor
    if name == "ridge":
        return Ridge(alpha=10.0)
    if name == "lasso":
        return Lasso(alpha=0.01, max_iter=2000)
    if name == "gbm":
        return GradientBoostingRegressor(random_state=random_state, **cfg["gbm"])
    if name == "rf":
        return RandomForestRegressor(random_state=random_state, n_jobs=1, **cfg["rf"])
    if name == "mlp":
        return MLPRegressor(random_state=random_state, **cfg["mlp"])
    raise ValueError(name)


def run_candidate(r, model, window, cfg, random_state):
    """
    Walk-forward one (model, window) candidate on the return series r.
    Expanding training window, refit every cfg['refit_every'] days.  At the close of day t the
    model sees r[t-window+1 .. t] and predicts r[t+1]; the position sign(prediction) is held
    over day t+1.  Training rows at a refit on day t0 are the feature days s <= t0-1, whose
    targets r[s+1] are all known by t0 -- nothing from the future is used.

    Returns (net, gross): daily strategy returns for days burn_in+1 .. end.
    `gross` ignores costs; `net` charges COST_PER_SIDE_BPS per unit of position change.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    from threadpoolctl import threadpool_limits
    n = len(r)
    B, K, w = cfg["burn_in"], cfg["refit_every"], window
    X_all = sliding_window_view(r, w)           # row i  <->  feature day t = i + w - 1
    pos = np.zeros(n)                           # pos[t]: decided at close of t, earns r[t+1]
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("ignore")
        for t0 in range(B, n - 1, K):
            t1 = min(t0 + K, n - 1)
            s = np.arange(w - 1, t0)
            y = r[s + 1]
            scale = y.std()
            m = _make_model(model, cfg, random_state)
            m.fit(X_all[s - (w - 1)] / scale, y / scale)
            t = np.arange(t0, t1)
            pos[t] = np.where(m.predict(X_all[t - (w - 1)] / scale) > 0.0, 1.0, -1.0)
    held = pos[B:n - 1]
    gross = held * r[B + 1:n]
    turnover = np.abs(np.diff(np.concatenate([[0.0], held])))
    net = gross - turnover * COST_PER_SIDE_BPS / 1e4
    return net, gross


def _candidate_job(args):
    return run_candidate(*args)


# ------------------------------------------------------------------ the search
def candidates():
    return list(product(MODELS, WINDOWS))


def run_search(seed=SEED, cfg=FULL, n_jobs=1):
    """Simulate the null world and run all 20 candidates.  Returns the (L x 20) return matrices."""
    n_returns = cfg["burn_in"] + 1 + cfg["is_days"] + cfg["oos_days"]
    price = simulate_null_world(n_returns, seed)
    r = price[1:] / price[:-1] - 1.0
    jobs = [(r, m, w, cfg, (seed + 7919 * i) % (2 ** 31)) for i, (m, w) in enumerate(candidates())]
    if n_jobs == 1:
        outs = [_candidate_job(j) for j in jobs]
    else:
        from joblib import Parallel, delayed
        outs = Parallel(n_jobs=n_jobs, backend="multiprocessing")(delayed(_candidate_job)(j) for j in jobs)
    return {"net": np.column_stack([o[0] for o in outs]),
            "gross": np.column_stack([o[1] for o in outs]),
            "returns": r, "seed": seed, "cfg": cfg}


def _ann_sharpes(M):
    return np.array([sharpe_ratio(M[:, j], PPY) for j in range(M.shape[1])])


def summarize(res, n_splits=16):
    """The search surface plus the toolkit's verdict on it."""
    cfg, net, gross = res["cfg"], res["net"], res["gross"]
    L = cfg["is_days"]
    cands = candidates()
    is_sr, oos_sr = _ann_sharpes(net[:L]), _ann_sharpes(net[L:])
    oos_gross = _ann_sharpes(gross[L:])
    years_is = L / PPY
    is_t = is_sr * np.sqrt(years_is)

    rep = analyze(net[:L], periods_per_year=PPY, n_splits=n_splits)   # the researcher's view
    best = rep["best_strategy"]
    oos_rank = int((oos_sr > oos_sr[best]).sum()) + 1                 # 1 = best OOS of the 20
    rho = spearmanr(is_sr, oos_sr)[0]
    surface = [dict(model=m, window=w, is_sharpe=is_sr[i], oos_sharpe=oos_sr[i], is_tstat=is_t[i])
               for i, (m, w) in enumerate(cands)]
    return {
        "surface": surface,
        "selection": dict(
            winner=f"{cands[best][0]}|w{cands[best][1]}", winner_index=best,
            winner_is_sharpe=is_sr[best], winner_oos_sharpe=oos_sr[best],
            winner_oos_sharpe_gross=oos_gross[best], winner_is_tstat=is_t[best],
            winner_oos_rank_of_20=oos_rank,
            n_candidates=len(cands), n_positive_is=int((is_sr > 0).sum()),
            n_is_tstat_gt2=int((is_t > 2).sum()), n_positive_oos=int((oos_sr > 0).sum()),
            mean_is_sharpe_all=is_sr.mean(), mean_oos_sharpe_all=oos_sr.mean(),
            mean_oos_sharpe_all_gross=oos_gross.mean(), spearman_is_vs_oos=rho),
        "toolkit": dict(
            n_trials=rep["n_strategies"], backtest_years=years_is,
            best_sharpe_annual=rep["best_sharpe_annual"], psr_vs_zero=rep["psr"], dsr=rep["dsr"],
            deflated_benchmark_sharpe_annual=rep["deflated_benchmark_sharpe_annual"],
            pbo=rep["pbo"], prob_oos_loss=rep["prob_oos_loss"], degradation=rep["degradation"],
            min_backtest_length_years=rep["min_backtest_length_years"],
            n_cscv_splits=n_splits, verdict=rep["verdict"]),
        "_report": rep,
    }


# ------------------------------------------------------------------ seed replication
def _replicate_one(seed, cfg, n_splits):
    res = run_search(seed, cfg, n_jobs=1)
    s = summarize(res, n_splits=n_splits)
    sel, tk = s["selection"], s["toolkit"]
    return dict(seed=seed, winner=sel["winner"], winner_is_sharpe=sel["winner_is_sharpe"],
                winner_oos_sharpe=sel["winner_oos_sharpe"],
                winner_oos_sharpe_gross=sel["winner_oos_sharpe_gross"],
                winner_is_tstat=sel["winner_is_tstat"],
                mean_oos_sharpe_all=sel["mean_oos_sharpe_all"],
                mean_oos_sharpe_all_gross=sel["mean_oos_sharpe_all_gross"],
                dsr=tk["dsr"], pbo=tk["pbo"], verdict=tk["verdict"])


def replicate(n_seeds, cfg=FAST, base_seed=SEED, n_jobs=1, n_splits=16):
    """The same null-world search repeated over n_seeds fresh seeds (base_seed + 1 ... + n_seeds)."""
    seeds = [base_seed + 1 + k for k in range(n_seeds)]
    if n_jobs == 1:
        return [_replicate_one(s, cfg, n_splits) for s in seeds]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(_replicate_one)(s, cfg, n_splits) for s in seeds)


def summarize_replication(recs, cfg):
    g = lambda k: np.array([x[k] for x in recs], dtype=float)
    n = len(recs)
    is_w, oos_w, oos_g = g("winner_is_sharpe"), g("winner_oos_sharpe"), g("winner_oos_sharpe_gross")
    dsr, pbo, t = g("dsr"), g("pbo"), g("winner_is_tstat")
    se = lambda x: float(x.std(ddof=1) / np.sqrt(len(x)))
    nominal = t > 2
    flagged = np.array(["survives" not in x["verdict"] for x in recs])
    return dict(
        config=cfg["name"], n_seeds=n,
        share_positive_is_winner=float((is_w > 0).mean()),
        mean_winner_is_sharpe=float(is_w.mean()),
        mean_winner_oos_sharpe=float(oos_w.mean()), se_winner_oos_sharpe=se(oos_w),
        mean_winner_oos_sharpe_gross=float(oos_g.mean()), se_winner_oos_sharpe_gross=se(oos_g),
        mean_all_candidates_oos_sharpe_gross=float(g("mean_oos_sharpe_all_gross").mean()),
        se_all_candidates_oos_sharpe_gross=se(g("mean_oos_sharpe_all_gross")),
        share_winner_oos_positive=float((oos_w > 0).mean()),
        share_nominal_winner_t_gt_2=float(nominal.mean()),
        share_dsr_below_095=float((dsr < 0.95).mean()), median_dsr=float(np.median(dsr)),
        mean_pbo=float(pbo.mean()), share_pbo_at_least_half=float((pbo >= 0.5).mean()),
        share_flagged_by_verdict=float(flagged.mean()),
        share_overfit_verdict=float(np.mean(["OVERFIT" in x["verdict"] for x in recs])),
        nominal_winners_rejected_by_dsr=(float((dsr[nominal] < 0.95).mean())
                                         if nominal.any() else None),
    )


# ------------------------------------------------------------------ output
def _r(x, k=4):
    if isinstance(x, dict):
        return {a: _r(b, k) for a, b in x.items() if not a.startswith("_")}
    if isinstance(x, (list, tuple)):
        return [_r(b, k) for b in x]
    if isinstance(x, (float, np.floating)):
        return round(float(x), k)
    if isinstance(x, np.integer):
        return int(x)
    return x


def make_figure(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    ink, ink2, muted, grid = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
    blue, orange, red = "#2a78d6", "#eb6834", "#e34948"
    div = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    plt.rcParams.update({"font.family": "sans-serif", "axes.edgecolor": "#c3c2b7",
                         "axes.labelcolor": ink2, "xtick.color": ink2, "ytick.color": ink2,
                         "text.color": ink, "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb"})

    surf, sel, tk = out["surface"], out["selection"], out["toolkit"]
    ncol = len(WINDOWS)
    IS = np.array([c["is_sharpe"] for c in surf]).reshape(len(MODELS), ncol)
    OOS = np.array([c["oos_sharpe"] for c in surf]).reshape(len(MODELS), ncol)
    lim = float(np.ceil(max(abs(IS).max(), abs(OOS).max()) * 10) / 10)
    wi = sel["winner_index"]
    wr, wc = divmod(wi, ncol)

    fig, ax = plt.subplots(2, 2, figsize=(12.5, 9.2))
    fig.suptitle("The best of 20 models on pure noise looks like a signal in the backtest; out of sample it is gone",
                 fontsize=14, fontweight="bold", x=0.02, ha="left", y=0.99)

    for a, M, ttl in ((ax[0, 0], IS, "In-sample (backtest) Sharpe"),
                      (ax[0, 1], OOS, "Out-of-sample (held-out) Sharpe")):
        a.imshow(M, cmap=div, vmin=-lim, vmax=lim, aspect="auto")
        a.set_xticks(range(ncol), [f"{w}d" for w in WINDOWS])
        a.set_yticks(range(len(MODELS)), MODELS)
        a.set_xlabel("lookback window")
        a.set_title(ttl, loc="left", fontsize=11, color=ink)
        for i in range(len(MODELS)):
            for j in range(ncol):
                dark = abs(M[i, j]) > 0.7 * lim
                a.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=9,
                       color="#ffffff" if dark else ink)
        a.add_patch(Rectangle((wc - .5, wr - .5), 1, 1, fill=False, ec=ink, lw=2.2))
        for s in a.spines.values():
            s.set_visible(False)
        a.tick_params(length=0)
    ax[0, 0].text(wc, wr - 0.62, "selected", ha="center", va="bottom", fontsize=8, color=ink)

    a = ax[1, 0]
    x, y = np.array([c["is_sharpe"] for c in surf]), np.array([c["oos_sharpe"] for c in surf])
    a.axhline(0, color="#c3c2b7", lw=1)
    a.axvline(0, color="#c3c2b7", lw=1)
    a.scatter(np.delete(x, wi), np.delete(y, wi), s=44, color=blue, ec="#fcfcfb", lw=1.2, zorder=3)
    a.scatter([x[wi]], [y[wi]], s=110, color=orange, ec="#fcfcfb", lw=1.5, zorder=4)
    slope, icpt = np.polyfit(x, y, 1)
    xs = np.array([x.min(), x.max()])
    a.plot(xs, icpt + slope * xs, color=muted, lw=1.5, ls="--", zorder=2)
    a.annotate(f"selected: {sel['winner']}\nIS {x[wi]:+.2f} -> OOS {y[wi]:+.2f}",
               (x[wi], y[wi]), xytext=(-40, 52), textcoords="offset points", ha="right",
               fontsize=9, color=ink, arrowprops=dict(arrowstyle="-", color=muted, lw=0.8))
    a.set_xlabel("in-sample Sharpe (annualised, net of costs)")
    a.set_ylabel("out-of-sample Sharpe")
    a.set_title(f"All 20 candidates: rank correlation IS vs OOS {sel['spearman_is_vs_oos']:+.2f}",
                loc="left", fontsize=11, color=ink)
    a.grid(color=grid, lw=0.6)
    a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)

    a = ax[1, 1]
    rep = out.get("replication")
    if rep:
        recs = rep["records"]
        i_s = np.array([q["winner_is_sharpe"] for q in recs])
        o_s = np.array([q["winner_oos_sharpe"] for q in recs])
        for u, v in zip(i_s, o_s):
            a.plot([0, 1], [u, v], color=blue, alpha=0.35, lw=1.2)
        a.scatter(np.zeros_like(i_s), i_s, s=26, color=blue, ec="#fcfcfb", lw=0.8, zorder=3)
        a.scatter(np.ones_like(o_s), o_s, s=26, color=blue, ec="#fcfcfb", lw=0.8, zorder=3)
        a.plot([0, 1], [i_s.mean(), o_s.mean()], color=orange, lw=3, zorder=4)
        a.scatter([0, 1], [i_s.mean(), o_s.mean()], s=90, color=orange, ec="#fcfcfb", lw=1.5, zorder=5)
        a.axhline(0, color="#c3c2b7", lw=1)
        a.text(-0.05, i_s.mean(), f"mean {i_s.mean():+.2f}", fontsize=9, color=ink, va="center", ha="right")
        a.text(1.05, o_s.mean(), f"mean {o_s.mean():+.2f}", fontsize=9, color=ink, va="center", ha="left")
        a.set_xticks([0, 1], ["selected in-sample", "same model, held-out"])
        a.set_xlim(-0.45, 1.45)
        a.set_ylabel("winner's annualised Sharpe")
        a.set_title(f"Repeated on {len(recs)} fresh seeds ({rep['summary']['config']} configuration)",
                    loc="left", fontsize=11, color=ink)
        a.grid(axis="y", color=grid, lw=0.6)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    else:
        a.axis("off")
        a.text(0.5, 0.5, "replication skipped", ha="center", va="center", color=muted)

    fig.text(0.02, 0.012,
             f"Null world: zero-mean GARCH(1,1) returns, seed {out['meta']['seed']}.  "
             f"Selected on the {tk['backtest_years']:.0f}-year backtest.  "
             f"DSR {tk['dsr']:.2f}, PBO {tk['pbo']:.2f}, "
             f"minimum backtest length for 20 trials {tk['min_backtest_length_years']:.1f} years.\n"
             "Heatmaps: red = positive Sharpe, blue = negative; all Sharpe ratios annualised and net of costs.",
             fontsize=9, color=ink2, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--replicate", type=int, default=30,
                    help="seeds for the FAST-configuration replication (0 = skip)")
    ap.add_argument("--n-jobs", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--out-dir", default=HERE)
    a = ap.parse_args()
    t0 = time.time()

    cands = candidates()
    print(f"\nCASE STUDY 5 -- ML model search on the null world "
          f"(seed {a.seed}, {len(cands)} candidates)\n")
    res = run_search(a.seed, FULL, n_jobs=a.n_jobs)
    s = summarize(res)
    sel, tk = s["selection"], s["toolkit"]
    print(f"  search finished in {time.time() - t0:.0f}s   "
          f"(backtest {FULL['is_days']} days, held-out {FULL['oos_days']} days)\n")
    print(f"  {'candidate':<10} {'IS Sharpe':>10} {'OOS Sharpe':>11} {'IS t':>7}")
    for c in sorted(s["surface"], key=lambda c: -c["is_sharpe"]):
        tag = "   <- selected" if f"{c['model']}|w{c['window']}" == sel["winner"] else ""
        print(f"  {c['model'] + '|w' + str(c['window']):<10} {c['is_sharpe']:>+10.2f} "
              f"{c['oos_sharpe']:>+11.2f} {c['is_tstat']:>+7.2f}{tag}")
    print()
    print(format_report(s["_report"]))

    out = {
        "meta": dict(
            study="case study 5: ML model search on the null world", seed=a.seed,
            world=dict(WORLD, mean="zero (by construction)"), models=list(MODELS),
            windows=list(WINDOWS), cost_bps_per_side=COST_PER_SIDE_BPS,
            config={k: (list(v) if isinstance(v, tuple) else v) for k, v in FULL.items()},
            note="IS = backtest used to pick the model; OOS = later held-out days. "
                 "Sharpe ratios are annualised (252 d) and net of costs unless marked gross."),
        "surface": s["surface"], "selection": sel, "toolkit": tk,
    }
    if a.replicate > 0:
        t1 = time.time()
        recs = replicate(a.replicate, FAST, base_seed=a.seed, n_jobs=a.n_jobs)
        summ = summarize_replication(recs, FAST)
        out["replication"] = dict(
            note="Same 20-candidate null-world search on the FAST configuration (shorter series, "
                 "lighter models, quarterly refits), fresh seeds.",
            config={k: (list(v) if isinstance(v, tuple) else v) for k, v in FAST.items()},
            summary=summ, records=recs)
        print(f"\n  replication: {a.replicate} fresh seeds ({time.time() - t1:.0f}s)")
        for k, v in summ.items():
            print(f"    {k:<42} {v if not isinstance(v, float) else round(v, 3)}")

    out = _r(out)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    make_figure(out, os.path.join(a.out_dir, "results.png"))
    print(f"\n  wrote results.json, results.png to {a.out_dir}   (total {time.time() - t0:.0f}s)\n")


if __name__ == "__main__":
    main()
