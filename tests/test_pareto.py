import pytest

from promptgate import GATE_REPEATS, PARETO_TOLERANCE, pareto_compare


def _rows(scores_by_fixture, metric="score"):
    """{fixture: [rep scores]} -> gate-style rows. One score gets a bare stem
    (repeats=1 shape); several get #rN suffixes."""
    rows = []
    for fixture, scores in scores_by_fixture.items():
        if len(scores) == 1:
            rows.append({"fixture": fixture, metric: scores[0]})
        else:
            for i, s in enumerate(scores, 1):
                rows.append({"fixture": f"{fixture}#r{i}", metric: s})
    return rows


def test_constants_match_production():
    assert GATE_REPEATS == 2
    assert PARETO_TOLERANCE == 1.0


def test_held_within_tolerance():
    r = pareto_compare(_rows({"a": [4, 4]}), _rows({"a": [3, 4]}), "score")
    assert r["per_fixture"]["a"]["verdict"] == "held"
    assert r["pareto_ok"] and not r["regressions"] and not r["wins"]


def test_exact_tolerance_is_held_not_regression():
    # A drop of exactly PARETO_TOLERANCE is judge-noise territory: must EXCEED.
    r = pareto_compare(_rows({"a": [4, 4]}), _rows({"a": [3, 3]}), "score")
    assert r["per_fixture"]["a"]["delta"] == -PARETO_TOLERANCE
    assert r["per_fixture"]["a"]["verdict"] == "held"
    assert r["pareto_ok"]


def test_regression_beyond_tolerance():
    r = pareto_compare(_rows({"a": [4, 4]}), _rows({"a": [2, 3]}), "score")
    assert r["per_fixture"]["a"]["verdict"] == "regression"
    assert not r["pareto_ok"] and r["regressions"] == ["a"]


def test_win_beyond_tolerance():
    r = pareto_compare(_rows({"a": [2, 3]}), _rows({"a": [4, 4]}), "score")
    assert r["per_fixture"]["a"]["verdict"] == "win"
    assert r["pareto_ok"] and r["wins"] == ["a"]


def test_aggregate_win_does_not_mask_slice_regression():
    # The scenario the guard exists for: the mean improves (3.0 -> 3.33) but
    # fixture c genuinely tanks. An aggregate-only gate passes this.
    cur = _rows({"a": [4, 4], "b": [2, 2], "c": [3, 3]})
    cand = _rows({"a": [4, 4], "b": [5, 5], "c": [1, 1]})
    r = pareto_compare(cur, cand, "score")
    assert r["wins"] == ["b"]
    assert r["regressions"] == ["c"]
    assert not r["pareto_ok"]


def test_single_rep_rows_without_suffix():
    r = pareto_compare(_rows({"a": [4]}), _rows({"a": [4]}), "score")
    assert r["per_fixture"]["a"]["verdict"] == "held"
    assert r["pareto_ok"]


def test_mismatched_fixture_sets_fail_closed():
    with pytest.raises(RuntimeError, match="fixture sets differ"):
        pareto_compare(_rows({"a": [4, 4], "b": [3, 3]}), _rows({"a": [4, 4]}), "score")
