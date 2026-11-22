"""promptgate: a human-approved prompt optimizer with a regression gate.

The loop:

    collect scored cases -> aggregate the judge's critiques (1 model call)
      -> propose a candidate prompt (1 model call, editable segments only)
      -> gate it head to head against the current prompt on frozen fixtures
         (repeats, per-fixture Pareto guard, deterministic checks)
      -> stage a strict winner as an UNLABELED version
      -> a human moves the production label, or nothing happens.

Fixtures are built by point-in-time replay (pit.py), so a case never sees
anything dated after the moment it is replaying.

Stdlib only. Every model call goes through an injected callable, so the tests
and the example run offline with stubs.
"""

from .bands import BANDS, band_for, band_pass, in_band, judge_regression
from .gate import consistency, gate
from .kappa import cohen_kappa, format_kappa, quadratic_weighted_kappa
from .optimizer import CycleResult, ModelBudget, run_cycle
from .pareto import GATE_REPEATS, PARETO_TOLERANCE, pareto_compare
from .pit import Barrier, LeakError, ReadOnlyViolation, build_inputs
from .store import PromptRegistry
from .surgery import PromptLayout, SurgeryError

__all__ = [
    "BANDS", "band_for", "band_pass", "in_band", "judge_regression",
    "consistency", "gate",
    "cohen_kappa", "format_kappa", "quadratic_weighted_kappa",
    "CycleResult", "ModelBudget", "run_cycle",
    "GATE_REPEATS", "PARETO_TOLERANCE", "pareto_compare",
    "Barrier", "LeakError", "ReadOnlyViolation", "build_inputs",
    "PromptRegistry",
    "PromptLayout", "SurgeryError",
]
