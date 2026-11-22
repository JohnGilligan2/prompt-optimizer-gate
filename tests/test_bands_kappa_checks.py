from promptgate import (band_for, band_pass, cohen_kappa, format_kappa, in_band,
                        judge_regression, quadratic_weighted_kappa)
from promptgate.checks import make_checks


def test_bands():
    assert [band_for(s) for s in (1, 2, 3, 4, 5)] == ["poor", "poor", "adequate", "good", "good"]
    assert in_band(4, "good") and not in_band(3, "good") and not in_band(True, "good")


def test_band_pass_floor_exact_unknown():
    assert band_pass({"q": 5}, {"q": "adequate"}, "floor") == (True, [])
    ok, misses = band_pass({"q": 5}, {"q": "adequate"}, "exact")
    assert ok is False and misses
    assert band_pass({}, {"q": "good"})[0] is None
    assert band_pass({"q": 1, "r": None}, {"q": "good", "r": "good"})[0] is False


def test_judge_regression_counts_a_missing_criterion_as_a_miss():
    cases = [{"case_id": "c1", "expect": {"q": "good", "r": "good"}}]
    out = judge_regression(cases, replays=2, judge_fn=lambda c: {"q": 5})
    assert out["passes"] == 0 and out["trials"] == 2
    assert "r=missing expected good" in out["rows"][0]["misses"]


def test_kappa():
    assert quadratic_weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 1.0
    assert quadratic_weighted_kappa([4], [4]) is None
    assert cohen_kappa(["good"] * 7, ["good"] * 7) is None    # the all-one-label trap
    assert format_kappa(None) == "n/a"


def test_checks():
    c = make_checks(banned_terms=["guaranteed"], max_words=5)
    assert c("plain text here") == []
    failed = set(c("Hi 100, this is \u201cguaranteed\u201d \u2014 as an AI"))
    assert failed >= {"no_em_en_dash", "no_smart_quotes", "no_ai_tells",
                      "no_banned_terms", "greeting_not_numeric", "within_length"}
