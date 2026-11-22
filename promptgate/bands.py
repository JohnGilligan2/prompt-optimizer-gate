"""Human verdicts as bands, and the check that the judge still lands in them.

A golden case carries a HUMAN reference verdict as a BAND per criterion rather
than a point score: 1-2 `poor`, 3 `adequate`, 4-5 `good`. Bands, because two
careful people will not agree that a note is a 4 rather than a 5, and freezing
that disagreement into a gate produces a gate that fails on taste. They will
agree it is good rather than poor.

Two uses:

  * `judge_regression` replays the JUDGE on the frozen artifacts the humans
    graded and demands every criterion land in the human's band. The rubric is
    text and text drifts; a guidance edit that makes the judge harsher shows up
    here as golden cases falling out of band, before it quietly bends the
    optimizer. Run it before trusting the prompt gate.
  * `band_pass` in floor mode scores a REGENERATED output: it must reach at
    least the bottom of the human's band for the original. An improvement is
    never a miss.

Unknown is never a pass anywhere in this file.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from .gate import consistency

BANDS = {"poor": (1, 2), "adequate": (3, 3), "good": (4, 5), "ungradable": (-1, -1)}


def band_for(score: int) -> str:
    for name, (lo, hi) in BANDS.items():
        if lo <= int(score) <= hi:
            return name
    return "ungradable"


def in_band(score: Any, band: str) -> bool:
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        return False
    lo, hi = BANDS.get(band, (-1, -1))
    return lo <= int(score) <= hi


def expected_bands(scores: dict) -> dict:
    """The human's point scores as bands; ungradable criteria are dropped."""
    return {k: band_for(v) for k, v in scores.items()
            if isinstance(v, int) and not isinstance(v, bool)}


def band_pass(scores: dict, bands: Optional[dict], mode: str = "floor"
              ) -> tuple[Optional[bool], list]:
    """True / False / None (unknown), and the misses.

    floor: at or above the BOTTOM of the human's band. exact: inside the band.
    A known miss is a fail even if another criterion is unknown; otherwise any
    unscorable criterion makes the whole trial unknown. Never fail-clean.
    """
    if not bands:
        return None, []
    misses, unknown = [], []
    for crit, band in bands.items():
        lo, hi = BANDS[band]
        s = scores.get(crit)
        if isinstance(s, bool) or not isinstance(s, (int, float)):
            unknown.append(crit)
            continue
        ok = (s >= lo) if mode == "floor" else (lo <= s <= hi)
        if not ok:
            misses.append(f"{crit}={s} expected {'>=' + str(lo) if mode == 'floor' else band}")
    if misses:
        return False, misses
    if unknown:
        return None, [f"{c} unscored" for c in unknown]
    return True, []


def judge_regression(cases: list, *, replays: int,
                     judge_fn: Callable[[dict], dict]) -> dict:
    """Replay the judge on every golden case, check exact bands, report pass^k.

    cases: [{"case_id", "expect": {criterion: band}, ...}]
    judge_fn(case) -> {criterion: score}
    """
    rows, passes, trials = [], 0, 0
    for c in cases:
        case_passes, misses = 0, []
        for _ in range(replays):
            scores = judge_fn(c)
            ok = True
            for key, band in c["expect"].items():
                if key not in scores or scores[key] is None:
                    # A criterion the human graded and the judge did not return
                    # is a MISS, never a silent skip.
                    ok = False
                    misses.append(f"{key}=missing expected {band}")
                    continue
                if not in_band(scores[key], band):
                    ok = False
                    misses.append(f"{key}={scores[key]} expected {band}")
            trials += 1
            if ok:
                passes += 1
                case_passes += 1
        rows.append({"case_id": c["case_id"], "expect": c["expect"], "passes": case_passes,
                     "replays": replays, "misses": sorted(set(misses))})
    out = consistency(passes, trials)
    out["rows"] = rows
    return out
