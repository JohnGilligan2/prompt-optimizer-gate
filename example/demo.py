"""The whole loop on a synthetic helpdesk, offline, with stub models.

    python example/demo.py

1. Check the judge against human grades before trusting it.
2. Build the gate fixtures point-in-time, and show what a naive replay leaks.
3. Cycle 1: the optimizer proposes a candidate that wins on average but
   wrecks one ticket. The Pareto guard fails it. Nothing is staged.
4. Cycle 2: a candidate that wins everywhere. Staged as an unlabeled version.
5. A human promotes it. That is the only way a prompt goes live.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

import world  # noqa: E402
from promptgate import (  # noqa: E402
    GATE_REPEATS, PARETO_TOLERANCE, PromptRegistry, format_kappa,
    quadratic_weighted_kappa, run_cycle,
)
from promptgate.checks import make_checks  # noqa: E402


def rule(title):
    print()
    print(title)
    print("-" * len(title))


def pareto_table(g):
    print(f"  {'fixture':<10} {'current':>8} {'candidate':>10} {'delta':>7}  verdict")
    for f, d in g["pareto"]["per_fixture"].items():
        print(f"  {f:<10} {d['current']:>8.2f} {d['candidate']:>10.2f} {d['delta']:>+7.2f}  {d['verdict']}")


def main():
    helpdesk = world.Helpdesk()
    checks = make_checks(banned_terms=["guaranteed"])

    rule("1. Is the judge worth trusting?")
    graded = [c for c in world.CASES if "human" in c]
    j = [c["score"] for c in graded]
    h = [c["human"] for c in graded]
    agree = sum(a == b for a, b in zip(j, h))
    print(f"  judge vs human on {len(graded)} synthetic cases: exact agreement {agree}/{len(graded)}, "
          f"weighted kappa {format_kappa(quadratic_weighted_kappa(j, h))}")

    rule("2. Fixtures, point in time")
    pit = world.fixtures(helpdesk)
    naive = world.fixtures(helpdesk, barrier=False)
    s = pit[0]["inputs"]
    print(f"  case-201 at its cutoff {s['cutoff']}: stage reads {s['target']['stage']!r} "
          f"(today it reads 'Solved'), {len(s['target']['messages'])} visible message(s)")
    print(f"  barrier dropped: {s['barrier_stats']}")
    cur = world.CURRENT_PROMPT
    for label, fx in (("naive replay", naive), ("point in time", pit)):
        scores = [world.judge(f, world.generate(cur, f["inputs"]), 1)["score"] for f in fx]
        print(f"  current prompt, {label:<13}: mean {sum(scores) / len(scores):.2f}  {scores}")
    print("  The naive replay reads case-206's own fix out of its future messages.")

    registry = PromptRegistry("ticket-note", cur)
    common = dict(generate=world.generate, judge=world.judge, checks=checks,
                  required_literals=[world.NOTE_HEADER])

    rule(f"3. Cycle 1 (repeats {GATE_REPEATS}, Pareto tolerance {PARETO_TOLERANCE} judge points)")
    r1 = run_cycle(world.CASES, registry, world.LAYOUT, pit,
                   model=world.StubModel([world.PROPOSAL_A]), **common)
    for line in r1.log:
        print("  " + line)
    print("  change notes: " + r1.change_notes)
    pareto_table(r1.gate_result)
    print(f"  exit code {r1.exit_code}; staged: {r1.staged_version}")

    rule("4. Cycle 2")
    r2 = run_cycle(world.CASES, registry, world.LAYOUT, pit,
                   model=world.StubModel([world.PROPOSAL_B]), **common)
    for line in r2.log:
        print("  " + line)
    pareto_table(r2.gate_result)
    c = r2.gate_result["consistency"]
    print(f"  pass^3 current {c['current']['pow_k'][3]:.2f} -> candidate {c['candidate']['pow_k'][3]:.2f}")
    print(f"  exit code {r2.exit_code}; production is still version {registry.get().version}")

    rule("5. The human step")
    registry.promote(r2.staged_version)
    print(f"  promoted version {r2.staged_version}; production is now version {registry.get().version}")
    print("  Rolling back is the same call with version 1. No deploy either way.")


if __name__ == "__main__":
    main()
