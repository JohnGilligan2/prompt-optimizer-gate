"""The per-fixture Pareto guard and the two constants that make it work.

An aggregate-mean gate can promote a candidate that got better on average
while genuinely regressing one slice of tickets (every billing fixture, say).
The average hides it. So the gate also compares each fixture head to head,
mean over repeats, candidate vs current, and fails any candidate that drops a
single fixture by more than PARETO_TOLERANCE, even when the overall mean
improves.
"""

from __future__ import annotations

# How many times the gate replays EACH fixture per arm. Generation and judging
# are nondeterministic. With few fixtures a single coin-flip case swings the
# mean: in production one fixture flipped between 2 and 4 across runs and
# produced a phantom win. Repeats damp that. Cost and time scale linearly.
GATE_REPEATS = 2

# Per-fixture no-regression tolerance, in judge points on the 1 to 5 scale.
# The judge is noisy: a single-repeat 2-to-4 flip moves a GATE_REPEATS=2
# fixture mean by a full 1.0, so a per-fixture drop must EXCEED this to count
# as a real regression rather than noise. If you want a tighter guard, raise
# GATE_REPEATS before lowering this.
PARETO_TOLERANCE = 1.0


def fixture_stem(name: str) -> str:
    """Collapse a repeat row key ('case-07#r2') back to its fixture stem."""
    return name.split("#r", 1)[0]


def pareto_compare(current_rows, candidate_rows, metric, tolerance=None):
    """Head to head PER FIXTURE, the complement to the aggregate mean.

        delta = candidate_fixture_mean - current_fixture_mean
        delta < -tolerance  -> "regression"  (real drop, beyond judge noise)
        delta > +tolerance  -> "win"         (real gain, beyond judge noise)
        otherwise           -> "held"

    Returns {"per_fixture": {stem: {current, candidate, delta, verdict}},
             "regressions": [...], "wins": [...], "pareto_ok": bool,
             "tolerance": float}. pareto_ok is True iff NO fixture regressed.

    Fails closed if the two arms did not run the same fixture set: a partial
    arm must never pass by comparing against nothing.
    """
    if tolerance is None:
        tolerance = PARETO_TOLERANCE

    def _means(rows):
        by = {}
        for r in rows:
            by.setdefault(fixture_stem(r["fixture"]), []).append(r[metric])
        return {f: sum(v) / len(v) for f, v in by.items()}

    cur, cand = _means(current_rows), _means(candidate_rows)
    if set(cur) != set(cand):
        raise RuntimeError(
            "pareto guard: fixture sets differ between arms (current-only: "
            f"{sorted(set(cur) - set(cand))}, candidate-only: "
            f"{sorted(set(cand) - set(cur))}); refusing to compare")

    per_fixture = {}
    for f in sorted(cur):
        delta = cand[f] - cur[f]
        if delta < -tolerance:
            verdict = "regression"
        elif delta > tolerance:
            verdict = "win"
        else:
            verdict = "held"
        per_fixture[f] = {"current": cur[f], "candidate": cand[f],
                          "delta": delta, "verdict": verdict}
    regressions = [f for f, d in per_fixture.items() if d["verdict"] == "regression"]
    wins = [f for f, d in per_fixture.items() if d["verdict"] == "win"]
    return {"per_fixture": per_fixture, "regressions": regressions,
            "wins": wins, "pareto_ok": not regressions, "tolerance": tolerance}
