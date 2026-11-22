"""The gate: current prompt vs candidate prompt, head to head on frozen fixtures.

PASS iff all of these hold:

  * every candidate output passes the deterministic checks,
  * candidate mean judge score >= current mean (ties pass),
  * the per-fixture Pareto guard holds: no single fixture's mean drops by more
    than PARETO_TOLERANCE (an aggregate win must not hide a regression on a
    slice of tickets),
  * every candidate output still contains each required literal. This is the
    un-skippable invariant check. In production it was a header that a
    downstream poller deduplicated on; a candidate that dropped it would have
    re-posted on every ticket, so it was rejected whatever it scored.

PASS decides whether the candidate is acceptable. Staging needs more: the
candidate must STRICTLY beat the current mean (`beats`). A tie passes the gate
and stays on disk.

The prompt under test is passed to `generate` as an argument, never poked into
a module global. The production version used an injection seam on the shipping
module and needed a `finally` to reset it, because a crashed gate could
otherwise leave a candidate prompt live in the process. An explicit argument
cannot leak that way.
"""

from __future__ import annotations

import hashlib
from typing import Callable, Iterable, Sequence

from .pareto import GATE_REPEATS, pareto_compare


def consistency(passes: int, trials: int, ks: Iterable[int] = (1, 3, 5)) -> dict:
    """Per-trial pass rate with pass^k and pass@k beside it.

    pass@k ("did it ever work in k tries") rises with k. pass^k ("did it work
    every time in k tries") falls with k, and it is the one a customer feels.
    None when there were no trials: unknown is not zero.
    """
    p = passes / trials if trials else None
    return {"passes": passes, "trials": trials, "per_trial": p,
            "pow_k": {k: (p ** k if p is not None else None) for k in ks},
            "at_k": {k: (1 - (1 - p) ** k if p is not None else None) for k in ks}}


def _run_arm(fixtures, prompt_text, *, generate, judge, checks, required_literals,
             repeats, pass_threshold):
    rows = []
    for rep in range(1, repeats + 1):
        for fx in fixtures:
            output = generate(prompt_text, fx["inputs"])
            verdict = judge(fx, output, rep)
            failures = list(checks(output)) if checks else []
            score = verdict.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                # Fail-unknown: a judge that returns no usable score is an
                # error for this run, never a silent zero or a silent skip.
                raise RuntimeError(
                    f"judge returned no numeric score for {fx['id']} (got {score!r})")
            rows.append({
                "fixture": fx["id"] if repeats == 1 else f"{fx['id']}#r{rep}",
                "assert_passed": not failures,
                "failures": failures,
                "score": score,
                "comment": verdict.get("comment", ""),
                "has_literals": all(lit in (output or "") for lit in required_literals),
                "trial_pass": (not failures) and score >= pass_threshold,
            })
    return rows


def gate(fixtures: Sequence[dict], current_text: str, candidate_text: str, *,
         generate: Callable[[str, dict], str],
         judge: Callable[[dict, str, int], dict],
         checks: Callable[[str], list] | None = None,
         required_literals: Sequence[str] = (),
         repeats: int = GATE_REPEATS,
         tolerance: float | None = None,
         pass_threshold: int = 4) -> dict:
    """Replay every fixture `repeats` times per arm and compare.

    fixtures: [{"id", "inputs", "reference", ...}], already point-in-time.
    generate(prompt_text, inputs) -> output text.
    judge(fixture, output, repeat_index) -> {"score": 1..5, "comment": str}.
    checks(output) -> list of failed check names (empty means clean).
    """
    if not fixtures:
        raise RuntimeError("no fixtures: refusing to gate against nothing")
    candidate_sha = hashlib.sha256(candidate_text.encode("utf-8")).hexdigest()[:12]
    arms = {}
    for label, text in (("current", current_text), ("candidate", candidate_text)):
        arms[label] = _run_arm(
            fixtures, text, generate=generate, judge=judge, checks=checks,
            required_literals=required_literals, repeats=repeats,
            pass_threshold=pass_threshold)

    def _mean(rows):
        return sum(r["score"] for r in rows) / len(rows)

    means = {label: _mean(rows) for label, rows in arms.items()}
    cand_asserts = all(r["assert_passed"] for r in arms["candidate"])
    metric_ok = means["candidate"] >= means["current"]
    pareto = pareto_compare(arms["current"], arms["candidate"], "score", tolerance)
    literals_ok = all(r["has_literals"] for r in arms["candidate"])
    passed = cand_asserts and metric_ok and pareto["pareto_ok"] and literals_ok
    beats = means["candidate"] > means["current"]

    reasons = []
    if not cand_asserts:
        bad = sorted({f for r in arms["candidate"] for f in r["failures"]})
        reasons.append("deterministic checks failed: " + ", ".join(bad))
    if not metric_ok:
        reasons.append(f"mean fell {means['current']:.2f} -> {means['candidate']:.2f}")
    if not pareto["pareto_ok"]:
        reasons.append("fixture regressed beyond tolerance: " + ", ".join(pareto["regressions"]))
    if not literals_ok:
        reasons.append("a candidate output dropped a required literal")

    return {
        "passed": passed,
        "beats": beats,
        "candidate_assertions_pass": cand_asserts,
        "metric_ok": metric_ok,
        "literals_ok": literals_ok,
        "pareto": pareto,
        "means": means,
        "consistency": {label: consistency(sum(r["trial_pass"] for r in rows), len(rows))
                        for label, rows in arms.items()},
        "arms": arms,
        "repeats": repeats,
        "candidate_sha": candidate_sha,
        "reasons": reasons,
    }
