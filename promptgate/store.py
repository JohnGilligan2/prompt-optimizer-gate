"""Prompt versions and labels. Staging is automatic; going live is not.

The production system kept prompts in a prompt-management service where the
generator only ever fetched the version carrying the `production` label. That
made an UNLABELED version inert, which is the whole safety model:

  * the optimizer may create a version (stage), with no labels;
  * only a human moves `production` onto it (promote), and moving it back is
    the rollback. No deploy either way.

`PromptRegistry` is an in-memory stand-in with the same contract. Swap in any
store that can create an unlabeled version and fetch by label.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PromptVersion:
    version: int
    text: str
    labels: set = field(default_factory=set)
    message: str = ""


class PromptRegistry:
    def __init__(self, name: str, initial_text: str):
        self.name = name
        self.versions = [PromptVersion(1, initial_text, {"production"}, "initial")]

    def get(self, label: str = "production") -> PromptVersion:
        for v in reversed(self.versions):
            if label in v.labels:
                return v
        raise LookupError(f"no version of {self.name!r} carries label {label!r}")

    def stage(self, text: str, message: str) -> int:
        """Create a new version with NO labels. Inert until a human promotes it."""
        v = PromptVersion(len(self.versions) + 1, text, set(), message)
        self.versions.append(v)
        return v.version

    def promote(self, version: int, label: str = "production") -> None:
        """The human step. Moves `label` onto `version` and off everything else."""
        target = next((v for v in self.versions if v.version == version), None)
        if target is None:
            raise LookupError(f"{self.name!r} has no version {version}")
        for v in self.versions:
            v.labels.discard(label)
        target.labels.add(label)
