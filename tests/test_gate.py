import pytest

from promptgate import consistency, gate
from promptgate.checks import make_checks

FIXTURES = [{"id": "f1", "inputs": {"n": 1}, "reference": {}},
            {"id": "f2", "inputs": {"n": 2}, "reference": {}}]


def gen(prompt, inputs):
    return f"[hdr] {prompt} {inputs['n']}"


def judge_by_prompt(table):
    """judge whose score is looked up by (prompt word, fixture id)."""
    def judge(fx, output, rep):
        word = output.split()[1]
        return {"score": table[(word, fx["id"])], "comment": ""}
    return judge


def test_tie_passes_but_does_not_beat():
    j = judge_by_prompt({("cur", "f1"): 4, ("cur", "f2"): 4, ("cand", "f1"): 4, ("cand", "f2"): 4})
    g = gate(FIXTURES, "cur", "cand", generate=gen, judge=j)
    assert g["passed"] and not g["beats"]


def test_strict_win_beats():
    j = judge_by_prompt({("cur", "f1"): 3, ("cur", "f2"): 3, ("cand", "f1"): 4, ("cand", "f2"): 4})
    g = gate(FIXTURES, "cur", "cand", generate=gen, judge=j)
    assert g["passed"] and g["beats"]
    assert len(g["arms"]["candidate"]) == 4          # 2 fixtures x 2 repeats
    assert g["arms"]["candidate"][0]["fixture"].endswith("#r1")


def test_slice_regression_fails_despite_better_mean():
    j = judge_by_prompt({("cur", "f1"): 1, ("cur", "f2"): 4, ("cand", "f1"): 5, ("cand", "f2"): 2})
    g = gate(FIXTURES, "cur", "cand", generate=gen, judge=j)
    assert g["means"]["candidate"] > g["means"]["current"]
    assert not g["passed"] and g["pareto"]["regressions"] == ["f2"]


def test_deterministic_check_failure_fails_gate():
    j = judge_by_prompt({("cur", "f1"): 3, ("cur", "f2"): 3, ("cand\u2014x", "f1"): 5,
                         ("cand\u2014x", "f2"): 5})
    g = gate(FIXTURES, "cur", "cand\u2014x", generate=gen, judge=j, checks=make_checks())
    assert not g["passed"] and not g["candidate_assertions_pass"]
    assert "no_em_en_dash" in g["reasons"][0]


def test_dropped_required_literal_fails_gate():
    def gen2(prompt, inputs):
        return ("[hdr] " if prompt == "cur" else "") + prompt
    j = lambda fx, out, rep: {"score": 5 if "cand" in out else 3}  # noqa: E731
    g = gate(FIXTURES, "cur", "cand", generate=gen2, judge=j, required_literals=["[hdr]"])
    assert g["beats"] and not g["passed"] and not g["literals_ok"]


def test_judge_without_a_score_is_an_error_not_a_zero():
    with pytest.raises(RuntimeError, match="no numeric score"):
        gate(FIXTURES, "cur", "cand", generate=gen, judge=lambda fx, o, r: {"score": None})


def test_no_fixtures_refused():
    with pytest.raises(RuntimeError, match="no fixtures"):
        gate([], "cur", "cand", generate=gen, judge=lambda fx, o, r: {"score": 5})


def test_consistency_pass_pow_k_and_unknown():
    c = consistency(9, 10)
    assert abs(c["pow_k"][5] - 0.9 ** 5) < 1e-12
    assert c["at_k"][5] > c["per_trial"] > c["pow_k"][5]
    assert consistency(0, 0)["per_trial"] is None
