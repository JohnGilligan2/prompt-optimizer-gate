import pytest

import world
from promptgate import ModelBudget, PromptRegistry, run_cycle
from promptgate.checks import make_checks
from promptgate.optimizer import build_corpus, usable_cases


def _run(proposals, registry=None):
    registry = registry or PromptRegistry("note", world.CURRENT_PROMPT)
    res = run_cycle(world.CASES, registry, world.LAYOUT, world.fixtures(),
                    model=world.StubModel(proposals), generate=world.generate,
                    judge=world.judge, checks=make_checks(),
                    required_literals=[world.NOTE_HEADER])
    return res, registry


def test_average_win_with_a_slice_regression_is_not_staged():
    res, reg = _run([world.PROPOSAL_A])
    g = res.gate_result
    assert g["means"]["candidate"] > g["means"]["current"]
    assert g["pareto"]["regressions"] == ["case-204"]
    assert res.exit_code == 2 and res.staged_version is None
    assert len(reg.versions) == 1


def test_clean_win_is_staged_unlabeled_and_production_is_untouched():
    res, reg = _run([world.PROPOSAL_B])
    assert res.exit_code == 0 and res.staged_version == 2
    assert reg.versions[1].labels == set()
    assert reg.get("production").version == 1


def test_cycle_never_promotes(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("run_cycle must never promote")
    monkeypatch.setattr(PromptRegistry, "promote", boom)
    res, _ = _run([world.PROPOSAL_B])
    assert res.staged_version == 2


def test_tie_passes_but_is_not_staged():
    pre, _mid, rules = world.LAYOUT.split(world.CURRENT_PROMPT)
    res, _reg = _run([{"preamble": pre, "rules": rules, "change_notes": "none"}])
    assert res.gate_result["passed"] and not res.gate_result["beats"]
    assert res.staged_version is None and res.exit_code == 0


def test_malformed_candidate_is_an_error_and_stages_nothing():
    bad = dict(world.PROPOSAL_B, rules="## RULES\n- whatever\n")
    res, reg = _run([bad])
    assert res.exit_code == 1 and "candidate proposal failed" in res.error
    assert res.gate_result is None and len(reg.versions) == 1


def test_gate_and_synthetic_cases_never_feed_the_corpus():
    ids = {c["id"] for c in usable_cases(world.CASES)}
    assert "g-01" not in ids and "t-01" not in ids
    corpus = build_corpus(usable_cases(world.CASES))
    assert "g-01" not in corpus and "t-01" not in corpus


def test_model_budget_is_a_hard_cap():
    b = ModelBudget(lambda s, u, sc: {}, max_calls=2)
    b("s", "u", {})
    b("s", "u", {})
    with pytest.raises(RuntimeError, match="hard cap"):
        b("s", "u", {})


def test_two_model_calls_per_cycle():
    model = world.StubModel([world.PROPOSAL_B])
    run_cycle(world.CASES, PromptRegistry("n", world.CURRENT_PROMPT), world.LAYOUT,
              world.fixtures(), model=model, generate=world.generate, judge=world.judge)
    assert len(model.calls) == 2
