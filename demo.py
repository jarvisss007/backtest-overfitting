"""
Demo: the same "impressive" backtest, judged honestly.

Scenario A — data mining: generate 200 strategies that are PURE NOISE (no edge whatsoever),
report the best one's Sharpe. It looks great. The tools should expose it: low Deflated Sharpe,
high Probability of Backtest Overfitting.

Scenario B — one genuinely profitable strategy among noise. The tools should let it through.

Run:  python demo.py
"""
import numpy as np
from overfit import analyze, format_report, min_backtest_length

rng = np.random.default_rng(7)
T, N = 1000, 200        # 1000 days, 200 tried strategies
PPY = 252


def scenario_a():
    print("\nSCENARIO A — 200 pure-noise strategies, cherry-pick the best\n")
    M = rng.normal(0.0, 0.01, size=(T, N))          # zero true edge
    rep = analyze(M, periods_per_year=PPY)
    print(format_report(rep))
    print(f"\n  The headline: 'best strategy Sharpe {rep['best_sharpe_annual']:.2f}' — "
          f"looks publishable.\n  The truth: it's the luckiest of {N} coin-flips.\n")


def scenario_b():
    print("\nSCENARIO B — one strong real edge (Sharpe ~3.0) hidden among 199 noise strategies\n")
    M = rng.normal(0.0, 0.01, size=(T, N))
    edge = 0.01 * 3.0 / np.sqrt(PPY)                 # daily mean for ~3.0 annual Sharpe
    M[:, 0] += edge                                  # strategy 0 has a true, strong edge
    rep = analyze(M, periods_per_year=PPY)
    print(format_report(rep))
    print(f"\n  Selected strategy #{rep['best_strategy']} "
          f"({'the real one — and it survives deflation' if rep['best_strategy'] == 0 else 'a noise one'}).")
    print("  Note: a *modest* real edge (Sharpe ~1.3) would be drowned by 200 noise trials —\n"
          "  which is itself the honest lesson. Only a strong edge clears the deflated bar.\n")


def rule_of_thumb():
    print("\nHOW LONG A BACKTEST YOU NEED (min backtest length, true edge = 0):\n")
    print(f"  {'# strategies tried':>20} {'to justify Sharpe 1.0':>24} {'Sharpe 2.0':>14}")
    for n in (10, 50, 200, 1000):
        print(f"  {n:>20} {min_backtest_length(n, 1.0):>21.1f} y "
              f"{min_backtest_length(n, 2.0):>11.1f} y")
    print("\n  i.e. the more configurations you try, the longer a backtest must be before a"
          "\n  given Sharpe means anything. Try 1000 things and 'Sharpe 1.0' needs decades.\n")


if __name__ == "__main__":
    scenario_a()
    scenario_b()
    rule_of_thumb()
