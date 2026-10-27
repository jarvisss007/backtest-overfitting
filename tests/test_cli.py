"""The command line: same numbers as analyze(), right column named, bad files refused with the reason."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from overfit import analyze, main  # noqa: E402


def write_csv(path, M, names, dates=True):
    with open(path, "w") as fh:
        fh.write(",".join((["date"] if dates else []) + names) + "\n")
        for i, row in enumerate(M):
            fh.write(",".join(([f"2020-01-{i % 28 + 1:02d}"] if dates else []) + [f"{x:.6f}" for x in row]) + "\n")
    return str(path)


def run(capsys, *args):
    main(list(args))
    return capsys.readouterr().out


def test_json_matches_analyze_and_names_the_column(tmp_path, capsys):
    rng = np.random.default_rng(3)
    M = np.round(rng.normal(0, 0.01, (640, 30)), 6)
    names = [f"cfg{j}" for j in range(30)]
    rep = json.loads(run(capsys, write_csv(tmp_path / "r.csv", M, names), "--json"))
    want = analyze(M)
    assert rep["best_strategy"] == want["best_strategy"] and rep["best_strategy_name"] == names[want["best_strategy"]]
    for k in ("dsr", "psr", "pbo", "best_sharpe_annual"):
        assert rep[k] == pytest.approx(want[k])


def test_a_real_edge_survives_and_is_named(tmp_path, capsys):
    rng = np.random.default_rng(7)
    M = rng.normal(0, 0.01, (1000, 40))
    M[:, 17] += 0.002                                   # about Sharpe 3 a year, hidden among 39 noise columns
    names = [f"noise{j}" for j in range(40)]
    names[17] = "edge"
    out = run(capsys, write_csv(tmp_path / "r.csv", M, names, dates=False))
    assert "VERDICT: survives" in out and "Best strategy column: edge" in out


@pytest.mark.parametrize("content, reason", [
    ("0.1,0.2\n0.3,0.4\n", "add a header row"),
    ("date,a\n2020-01-01,0.1\n2020-01-02,0.2\n", "at least 2 strategy columns"),
    ("a,b\n0.1,0.2\n0.3,\n", "line 3 has an empty or non-numeric value"),
    ("a,b\n0.1,0.2\n0.3\n", "line 3 has 1 values"),
])
def test_bad_files_are_refused_with_the_reason(tmp_path, content, reason):
    p = tmp_path / "bad.csv"
    p.write_text(content)
    with pytest.raises(SystemExit) as e:
        main([str(p)])
    assert reason in str(e.value)


def test_too_few_rows_for_the_splits(tmp_path):
    p = write_csv(tmp_path / "short.csv", np.zeros((20, 3)) + 0.001, ["a", "b", "c"])
    with pytest.raises(SystemExit, match="too few for 16 CSCV splits"):
        main([p])
