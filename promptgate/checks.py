"""Deterministic checks: house rules as code, run beside the judge.

Milliseconds, no model call. They catch the class of failure that is not a
judgment call (dashes, smart quotes, AI tells, a banned term, a greeting that
picked up a number) and they cannot be charmed: a candidate that learns to
please the judge still has to pass them.

`make_checks` returns a callable for the gate: output -> list of failed names.
"""

from __future__ import annotations

import re
from typing import Iterable

_DASH = re.compile("[\u2014\u2013]")          # em / en dash
_SMART_Q = re.compile("[\u201c\u201d\u2018\u2019]")
_AI_TELLS = re.compile(r"\b(as an AI|AI[- ]generated|language model|I am an AI)\b", re.I)
_GREETING = re.compile(r"\b(hi|hello|dear)\s+([^\s,<]+)", re.I)


def make_checks(banned_terms: Iterable[str] = (), max_words: int | None = None):
    banned = [t for t in banned_terms if t]
    banned_rx = (re.compile(r"\b(" + "|".join(map(re.escape, banned)) + r")\b", re.I)
                 if banned else None)

    def checks(output: str) -> list:
        text = output or ""
        failed = []
        if _DASH.search(text):
            failed.append("no_em_en_dash")
        if _SMART_Q.search(text):
            failed.append("no_smart_quotes")
        if _AI_TELLS.search(text):
            failed.append("no_ai_tells")
        if banned_rx and banned_rx.search(text):
            failed.append("no_banned_terms")
        g = _GREETING.search(text)
        if g and g.group(2).isdigit():
            failed.append("greeting_not_numeric")
        if max_words is not None and len(text.split()) > max_words:
            failed.append("within_length")
        return failed

    return checks
