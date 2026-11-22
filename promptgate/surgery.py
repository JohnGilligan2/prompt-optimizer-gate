"""Prompt surgery: the model rewrites only the editable segments.

The production prompt had four parts, in order:

    1. preamble                          (editable)
    2. company technical context         (immutable slab)
    3. voice and tone, with exemplars    (immutable slab)
    4. rules, from a fixed heading to the end (editable)

The proposer only ever sees and rewrites segments 1 and 4. Code reassembles
the candidate with the original middle slab byte-identical, so the model
cannot "helpfully" paraphrase the exemplars or the context. Anchor headings
are validated so a malformed candidate fails closed before it costs a single
gate call.

The validator checks for gutting as well as for format: a rules segment that
lost a required literal, a hard-rule family, or more than half its length is
refused. In production one required literal was a header a downstream poller
deduplicated on, so losing it was not cosmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class SurgeryError(RuntimeError):
    """The current prompt or the candidate does not have the expected shape."""


@dataclass(frozen=True)
class PromptLayout:
    immutable_anchor: str           # first heading of the immutable middle
    rules_anchor: str               # heading that starts the editable rules
    immutable_headings: tuple = ()  # other headings inside the middle slab
    required_literals: tuple = ()   # must survive verbatim in the rules
    required_fragments_ci: tuple = field(default=())  # rule families, case-insensitive
    min_rules_ratio: float = 0.5
    min_preamble_ratio: float = 0.4
    max_growth: float = 1.6

    def split(self, text: str) -> tuple:
        """(preamble, immutable_middle, rules). Fails closed on missing or
        out-of-order anchors."""
        i_mid = text.find(self.immutable_anchor)
        i_rules = text.find(self.rules_anchor)
        if i_mid < 0 or i_rules < 0 or not i_mid < i_rules:
            raise SurgeryError(
                "prompt anchors not found in the expected order "
                f"(immutable={i_mid}, rules={i_rules}); refusing to edit")
        return text[:i_mid], text[i_mid:i_rules], text[i_rules:]

    @staticmethod
    def reassemble(preamble: str, middle: str, rules: str) -> str:
        return preamble.rstrip("\n") + "\n\n" + middle + rules.lstrip("\n")

    def check_round_trip(self, text: str) -> tuple:
        parts = self.split(text)
        if self.reassemble(*parts) != text:
            raise SurgeryError(
                "split/reassemble round trip changed the CURRENT prompt; its shape "
                "drifted, update the layout before running")
        return parts

    def validate(self, preamble_new: str, rules_new: str,
                 preamble_old: str, rules_old: str) -> None:
        problems = []
        if not rules_new.lstrip().startswith(self.rules_anchor):
            problems.append(f'rules segment must start with the exact heading "{self.rules_anchor}"')
        for name, seg in (("preamble", preamble_new), ("rules", rules_new)):
            for heading in (self.immutable_anchor, *self.immutable_headings):
                if heading in seg:
                    problems.append(f'{name} reproduces the immutable heading "{heading}" '
                                    "(the slab is reinserted by code)")
        for lit in self.required_literals:
            if lit not in rules_new:
                problems.append(f'rules segment lost the required literal "{lit}"')
        low = rules_new.lower()
        for frag in self.required_fragments_ci:
            if frag.lower() not in low:
                problems.append(f'rules segment no longer mentions "{frag}"; a hard rule '
                                "family looks deleted")
        if len(rules_new) < self.min_rules_ratio * len(rules_old):
            problems.append("rules segment shrank by more than half; refusing (looks gutted)")
        if len(preamble_new) < self.min_preamble_ratio * len(preamble_old):
            problems.append("preamble shrank by more than 60 percent; refusing (looks gutted)")
        if problems:
            raise SurgeryError("candidate rejected before the gate:\n  - " + "\n  - ".join(problems))

    def apply(self, current: str, preamble_new: str, rules_new: str) -> str:
        """Validate the model's segments and rebuild the full candidate."""
        preamble, middle, rules = self.check_round_trip(current)
        self.validate(preamble_new, rules_new, preamble, rules)
        candidate = self.reassemble(preamble_new, middle, rules_new)
        if len(candidate) > self.max_growth * len(current):
            raise SurgeryError(f"candidate is more than {self.max_growth}x the current prompt; "
                               "refusing (likely slab duplication or rule sprawl)")
        return candidate
