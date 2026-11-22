"""Chance-corrected agreement, because raw agreement flatters the judge.

"The judge agreed with the human on N of M verdicts" has a known defect: two
graders who both score most cases "about a 4" will agree often by accident,
and raw agreement cannot tell accident from skill. Report kappa beside it.

WHICH KAPPA. Cohen's kappa with QUADRATIC weights, the standard for ordinal
scales. Unweighted kappa treats a 4-vs-5 disagreement exactly like a 1-vs-5
one, which throws away the thing the scale is for.

    kappa_w = 1 - sum(w_ij * O_ij) / sum(w_ij * E_ij)

    O_ij  observed count of (judge=i, human=j)
    E_ij  expected count under independence, row_i * col_j / N
    w_ij  (i - j)^2 / (K - 1)^2, K the number of categories

WHAT IT DOES NOT TELL YOU. A judge that is a constant ONE point harsh still
loses kappa, because kappa asks "how far apart" and not "is the gap
consistent". A constant offset is fixable with one rubric edit. So read the
direction of the misses as well; kappa is the cross-check that agreement is not
an accident of clustering.

WHEN IT IS UNDEFINED. Fewer than two pairs, or marginals that leave no room
for chance disagreement. The functions return None and format_kappa prints
"n/a" rather than a zero, because a zero is a claim.
"""
from __future__ import annotations

from collections import Counter
from typing import Sequence

SCALE_MIN, SCALE_MAX = 1, 5
CATEGORIES = tuple(range(SCALE_MIN, SCALE_MAX + 1))


def quadratic_weighted_kappa(judge: Sequence[int], human: Sequence[int], *,
                             categories: Sequence[int] = CATEGORIES
                             ) -> float | None:
    """Cohen's kappa with quadratic weights over paired ordinal scores.

    Values outside `categories` are clamped to the nearest end rather than
    dropped: a clamp is a visible rounding, a drop would silently shrink N.
    """
    if len(judge) != len(human):
        raise ValueError("judge and human must be the same length")
    n = len(judge)
    if n < 2:
        return None

    lo, hi = min(categories), max(categories)
    span = float(hi - lo)
    if span <= 0:
        return None

    def clamp(v: int) -> int:
        return max(lo, min(hi, int(v)))

    pairs = Counter((clamp(j), clamp(h)) for j, h in zip(judge, human))
    row: Counter = Counter()
    col: Counter = Counter()
    for (j, h), c in pairs.items():
        row[j] += c
        col[h] += c

    def w(i: int, j: int) -> float:
        return ((i - j) ** 2) / (span ** 2)

    observed = sum(w(j, h) * c for (j, h), c in pairs.items())
    expected = sum(w(j, h) * row[j] * col[h] / n for j in row for h in col)
    if expected <= 0:
        return None
    kappa = 1.0 - observed / expected
    # A judge that says the same thing whatever the human says lands on exactly
    # zero in exact arithmetic and on 2e-16 in floating point. Snap it.
    return 0.0 if abs(kappa) < 1e-12 else kappa


def cohen_kappa(a: Sequence, b: Sequence) -> float | None:
    """Unweighted Cohen's kappa over paired nominal labels (a yes/no call).

        kappa = (p_o - p_e) / (1 - p_e)

    None when undefined: fewer than two pairs, or p_e == 1, which happens when
    BOTH raters used one and the same label for every case. Raw agreement is
    100 percent there and says nothing, so it is reported as undefined, never
    as 1.0.
    """
    if len(a) != len(b):
        raise ValueError("a and b must be the same length")
    n = len(a)
    if n < 2:
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    if pe >= 1.0 - 1e-12:
        return None
    kappa = (po - pe) / (1.0 - pe)
    return 0.0 if abs(kappa) < 1e-12 else kappa


def landis_koch(kappa: float | None) -> str:
    """The conventional band for a kappa. A label, not a verdict."""
    if kappa is None:
        return "n/a"
    if kappa < 0:
        return "worse than chance"
    if kappa <= 0.20:
        return "slight"
    if kappa <= 0.40:
        return "fair"
    if kappa <= 0.60:
        return "moderate"
    if kappa <= 0.80:
        return "substantial"
    return "almost perfect"


def format_kappa(kappa: float | None) -> str:
    if kappa is None:
        return "n/a"
    return f"{kappa:.2f} ({landis_koch(kappa)})"
