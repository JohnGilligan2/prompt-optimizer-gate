"""One optimization cycle: critiques in, at most one staged candidate out.

    cases -> aggregate_weaknesses (1 model call)
          -> propose_candidate    (1 model call, editable segments only)
          -> gate                 (generator + judge calls on frozen fixtures)
          -> stage if PASS and a strict beat, as an unlabeled version

Nothing here can make a prompt live. `run_cycle` never calls
`PromptRegistry.promote`; a test pins that.

`model(system, user, schema) -> dict` is the only door to an LLM for the two
optimizer calls. It is wrapped in ModelBudget, a hard cap in code, so a bug in
a retry path cannot turn one weekly run into a hundred expensive calls.

Exit codes follow the production CLI: 0 the cycle ran and the candidate passed
its gate, 1 an error (fail closed, nothing staged), 2 the cycle ran and the
candidate failed its gate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Optional

from .gate import gate as run_gate
from .pareto import GATE_REPEATS
from .store import PromptRegistry
from .surgery import PromptLayout


class ModelBudget:
    """Hard cap on optimizer model calls. Two per cycle: aggregate + propose."""

    def __init__(self, model: Callable[[str, str, dict], dict], max_calls: int = 2):
        self._model = model
        self.max_calls = max_calls
        self.calls = 0

    def __call__(self, system: str, user: str, schema: dict) -> dict:
        if self.calls >= self.max_calls:
            raise RuntimeError(f"hard cap exceeded: more than {self.max_calls} optimizer model calls")
        self.calls += 1
        return self._model(system, user, schema)


WEAKNESS_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "weaknesses": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "theme": {"type": "string"},
                    "evidence": {"type": "string"},
                    "suggested_rule_change": {"type": "string"},
                },
                "required": ["theme", "evidence", "suggested_rule_change"],
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["weaknesses", "summary"],
}

CANDIDATE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "preamble": {"type": "string"},
        "rules": {"type": "string"},
        "change_notes": {"type": "string"},
    },
    "required": ["preamble", "rules", "change_notes"],
}

WEAKNESS_SYSTEM = """You are the prompt-optimization analyst for a helpdesk ticket assistant.

An LLM judge scored past outputs 1-5 with a one-line comment. You get the score log and the K lowest-scoring outputs.

Identify RECURRING failure modes attributable to the PROMPT itself (instructions that are missing, ambiguous, or too weak), not one-off data problems or judge noise.

Rules:
- Report a weakness only when at least two cases support it, or one case shows a severe rule breach.
- evidence: cite the case ids and judge comments the weakness rests on.
- suggested_rule_change: a concrete, minimal prompt edit (add or TIGHTEN a rule). Never suggest loosening or removing a safety rule.
- Order weaknesses by expected impact, at most 6.
- If there is no real recurring weakness, return an empty list and say so in summary."""

CANDIDATE_SYSTEM = """You are revising the SYSTEM PROMPT of a helpdesk ticket assistant.

The prompt has three parts: an editable PREAMBLE, an IMMUTABLE middle section that you are not shown (code reinserts it verbatim), and editable RULES that start at a fixed heading.

Return the FULL text of both editable segments. If a segment needs no change, return it unchanged.
- The rules segment MUST still begin with its exact heading. Do not rename headings.
- Do not reproduce or restate the immutable section.
- Keep every existing hard rule unless a listed weakness justifies TIGHTENING it.
- Prefer minimal edits: every change must trace to a listed weakness.
- change_notes: one line per change, naming the edit and the weakness behind it."""


def usable_cases(cases: list) -> list:
    """Cases the optimizer may learn from.

    Dropped: anything without a numeric score; synthetic or test invocations;
    and the gate's own runs (source == "gate"), so a gate replay can never feed
    back into the corpus that proposes the next candidate.
    """
    out = []
    for c in cases:
        if c.get("source") == "gate" or c.get("synthetic"):
            continue
        s = c.get("score")
        if isinstance(s, bool) or not isinstance(s, (int, float)):
            continue
        out.append(c)
    return out


def _clip(text, n):
    text = (text or "").strip()
    return text if len(text) <= n else text[:n] + " ...[clipped]"


def build_corpus(cases: list, k_lowest: int = 5, max_cases: int = 60) -> str:
    """The score log plus the K lowest-scoring outputs, clipped."""
    scored = sorted(cases, key=lambda c: c.get("timestamp", ""), reverse=True)
    lines = [f"SCORED CASES (newest first, {len(scored)} total):"]
    for c in scored[:max_cases]:
        lines.append(f"- case {c['id']} @ {c.get('timestamp', '')[:19]} score={c['score']}")
        if c.get("comment"):
            lines.append(f'    judge: "{_clip(c["comment"], 200)}"')
    k = min(k_lowest, len(scored))
    lowest = sorted(scored, key=lambda c: c["score"])[:k]
    lines += ["", f"LOWEST-SCORING EXAMPLES (K={k}):"]
    for c in lowest:
        lines += ["", f"=== case {c['id']} score={c['score']} ===",
                  f'JUDGE: "{_clip(c.get("comment"), 300)}"',
                  "OUTPUT (clipped):", _clip(c.get("output") or "(none recorded)", 1500)]
    return "\n".join(lines)


def aggregate_weaknesses(cases: list, model: Callable, k_lowest: int = 5) -> dict:
    scored = usable_cases(cases)
    if not scored:
        raise RuntimeError("no usable scored cases: nothing to aggregate")
    return model(WEAKNESS_SYSTEM, build_corpus(scored, k_lowest), WEAKNESS_SCHEMA)


def propose_candidate(current_text: str, weaknesses: dict, layout: PromptLayout,
                      model: Callable) -> tuple:
    """One model call producing revised editable segments; code rebuilds the rest."""
    preamble, _middle, rules = layout.check_round_trip(current_text)
    user = ("OBSERVED WEAKNESSES:\n" + json.dumps(weaknesses, indent=2)
            + "\n\nCURRENT PREAMBLE:\n<<<PREAMBLE\n" + preamble + "\nPREAMBLE>>>"
            + "\n\nCURRENT RULES:\n<<<RULES\n" + rules + "\nRULES>>>")
    out = model(CANDIDATE_SYSTEM, user, CANDIDATE_SCHEMA)
    candidate = layout.apply(current_text, out["preamble"], out["rules"])
    return candidate, out.get("change_notes", "")


@dataclass
class CycleResult:
    exit_code: int
    weaknesses: Optional[dict] = None
    candidate_text: Optional[str] = None
    change_notes: str = ""
    gate_result: Optional[dict] = None
    staged_version: Optional[int] = None
    error: Optional[str] = None
    log: list = field(default_factory=list)


def run_cycle(cases: list, registry: PromptRegistry, layout: PromptLayout,
              fixtures: list, *, model: Callable, generate: Callable, judge: Callable,
              checks: Optional[Callable] = None, required_literals=(),
              repeats: int = GATE_REPEATS, tolerance: Optional[float] = None,
              max_model_calls: int = 2) -> CycleResult:
    budget = ModelBudget(model, max_model_calls)
    res = CycleResult(exit_code=1)

    try:
        res.weaknesses = aggregate_weaknesses(cases, budget)
    except Exception as e:  # noqa: BLE001 - fail closed: report the error, stage nothing
        res.error = f"weakness aggregation failed: {e}"
        return res
    res.log.append(f"{len(res.weaknesses.get('weaknesses', []))} weakness(es): "
                   + res.weaknesses.get("summary", ""))

    current = registry.get("production")
    try:
        res.candidate_text, res.change_notes = propose_candidate(
            current.text, res.weaknesses, layout, budget)
    except Exception as e:  # noqa: BLE001 - a malformed candidate is an error, not a FAIL
        res.error = f"candidate proposal failed: {e}"
        return res

    try:
        g = run_gate(fixtures, current.text, res.candidate_text, generate=generate,
                     judge=judge, checks=checks, required_literals=required_literals,
                     repeats=repeats, tolerance=tolerance)
    except Exception as e:  # noqa: BLE001 - a gate that cannot run never passes
        res.error = f"gate failed to run: {e}"
        return res
    res.gate_result = g
    m = g["means"]
    res.log.append(f"gate {'PASS' if g['passed'] else 'FAIL'}: mean {m['current']:.2f} -> "
                   f"{m['candidate']:.2f}" + ("" if g["passed"] else "; " + "; ".join(g["reasons"])))

    if g["passed"] and g["beats"]:
        res.staged_version = registry.stage(
            res.candidate_text,
            f"optimizer candidate (gate PASS, mean {m['current']:.2f} -> {m['candidate']:.2f}, "
            f"sha {g['candidate_sha']})")
        res.log.append(f"staged as version {res.staged_version}, unlabeled; "
                       "a human moves the production label to make it live")
    elif g["passed"]:
        res.log.append("gate PASS but only tied the current mean: not staged")
    res.exit_code = 0 if g["passed"] else 2
    return res
