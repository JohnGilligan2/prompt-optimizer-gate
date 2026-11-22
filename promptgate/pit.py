"""Point-in-time replay: show the generator the helpdesk as it was at the cutoff.

A golden case is a ticket at one moment, usually the moment before an engineer
worked it, with a reference answer taken from what happened AFTER that moment.
Replaying today's prompt on today's helpdesk leaks the answer in at least four
ways, and every one of them inflates the score:

  * the ticket's own later messages, which contain the fix;
  * tickets created after the cutoff, which retrieval happily returns because
    the search index answers with CURRENT data (a repeat of the same fault next
    month carries the exact fix);
  * the messages of older similar tickets that were written after the cutoff;
  * tracked fields at their FINAL value: the target's stage reads "Solved", and
    every similar ticket shows how it ended.

So everything the generator sees passes through `Barrier`, a read-only filter
between the retrieval code and the helpdesk:

  * messages: only those dated strictly before the cutoff, and never the
    reference's evidence messages (excluded by id as well as by date, so the
    answer stays out even if a cutoff were ever set late);
  * tickets: the target itself and every ticket created at or after the cutoff
    are dropped from every search;
  * stage and assignee on every ticket are rewound to their value at the cutoff
    from the change log, and the last-updated timestamp is blanked;
  * anything that is not a read raises ReadOnlyViolation. A replay cannot post.

Then `audit_inputs` re-checks the assembled inputs and raises LeakError if any
message, ticket or self-reference got through. Belt and braces: the barrier is
the mechanism, the audit is the proof.

The helpdesk is anything with these read methods (duck-typed):

    search_tickets(text, limit) -> [ticket]
    get_ticket(ticket_id)       -> ticket | None
    get_messages(ticket_id)     -> [message]
    read_messages(ids)          -> [message]
    tracking_changes(ids)       -> {ticket_id: {field: [(date, old, new), ...]}}

ticket:  {"id", "subject", "customer", "created", "stage", "assignee", "updated"}
message: {"id", "ticket_id", "date", "author", "body"}

Timestamps are ISO strings in one format ("YYYY-MM-DDTHH:MM:SS") so they
compare as strings; `norm_dt` normalizes.
"""

from __future__ import annotations

import copy
from typing import Any, Iterable, Optional

AS_OF_FIELDS = ("stage", "assignee")


class LeakError(RuntimeError):
    """Something dated at or after the cutoff reached the generator's inputs."""


class ReadOnlyViolation(RuntimeError):
    """A replay tried to write to the helpdesk."""


class ReferenceDefect(LookupError):
    """The golden case itself is broken (missing or foreign evidence). An error
    trial, never a pass."""


def norm_dt(s: Any) -> str:
    return str(s or "").replace(" ", "T")[:19]


def value_as_of(changes: list, cutoff: str, current: Any) -> Any:
    """Value of a tracked field at `cutoff`, from (date, old, new) changes.

    The last change strictly before the cutoff gives its NEW value; failing
    that, the first change at or after it gives its OLD value; with no changes
    at all the field never moved and the current value was the value then.
    """
    ordered = sorted(changes, key=lambda c: c[0])
    before = [c for c in ordered if c[0] < cutoff]
    if before:
        return before[-1][2]
    if ordered:
        return ordered[0][1]
    return current


class Barrier:
    """Read-only view of a helpdesk as it was at `cutoff`, minus the target."""

    # How many extra search hits to ask for, because the date filter runs
    # after the search and would otherwise starve retrieval.
    WIDEN = 10

    def __init__(self, inner: Any, *, cutoff: str, target_id: Any,
                 exclude_message_ids: Optional[Iterable] = None):
        self._inner = inner
        self.cutoff = norm_dt(cutoff)
        self.target_id = target_id
        self.exclude = set(exclude_message_ids or [])
        self._changes: dict = {}
        self.stats = {"tickets_dropped": 0, "messages_dropped": 0,
                      "evidence_dropped": 0, "as_of_rewrites": 0}

    def __getattr__(self, name):
        # Only the read methods below exist. Anything else (post_message,
        # update_ticket, ...) is refused rather than passed through.
        raise ReadOnlyViolation(f"point-in-time replay refused {name!r}: reads only")

    # -- reads ---------------------------------------------------------------
    def search_tickets(self, text: str, limit: int) -> list:
        rows = self._inner.search_tickets(text, limit + self.WIDEN)
        kept = [copy.deepcopy(r) for r in rows if self._ticket_visible(r)]
        self.stats["tickets_dropped"] += len(rows) - len(kept)
        return self.rewind(kept)[:limit]

    def get_ticket(self, ticket_id) -> Optional[dict]:
        row = self._inner.get_ticket(ticket_id)
        if row is None or not self._ticket_visible(row):
            return None
        return self.rewind([copy.deepcopy(row)])[0]

    def get_messages(self, ticket_id) -> list:
        return self._filter_messages(self._inner.get_messages(ticket_id))

    def read_messages(self, ids) -> list:
        return self._filter_messages(self._inner.read_messages(ids))

    # -- filtering -----------------------------------------------------------
    def _ticket_visible(self, row: dict) -> bool:
        return (row.get("id") != self.target_id
                and bool(row.get("created")) and norm_dt(row["created"]) < self.cutoff)

    def _filter_messages(self, rows: list) -> list:
        kept = []
        for m in rows:
            if m.get("id") in self.exclude:
                self.stats["evidence_dropped"] += 1
                continue
            if not m.get("date") or norm_dt(m["date"]) >= self.cutoff:
                self.stats["messages_dropped"] += 1
                continue
            kept.append(copy.deepcopy(m))
        return kept

    def rewind(self, rows: list) -> list:
        """stage / assignee to their as-of-cutoff values; `updated` blanked."""
        need = [r["id"] for r in rows if r["id"] not in self._changes]
        if need:
            got = self._inner.tracking_changes(need)
            for i in need:
                self._changes[i] = got.get(i, {})
        for r in rows:
            ch = self._changes.get(r["id"], {})
            for f in AS_OF_FIELDS:
                if f in r:
                    new = value_as_of(ch.get(f, []), self.cutoff, r[f])
                    if new != r[f]:
                        self.stats["as_of_rewrites"] += 1
                    r[f] = new
            if "updated" in r:
                r["updated"] = None
        return rows


def audit_evidence(helpdesk: Any, case: dict, cutoff: str) -> dict:
    """Check the reference's evidence messages, read through the UNFILTERED helpdesk.

    Each must exist and belong to this ticket. One dated AT OR AFTER the cutoff
    is the answer and stays hidden by id. One dated BEFORE it was visible at
    the cutoff (typically the customer's own report, cited for the symptom); it
    moves to `context_evidence_ids` and stays in the input, because hiding it
    would starve the generator of the question itself. Missing or foreign
    evidence is a ReferenceDefect.
    """
    ids = list(case.get("evidence_ids") or [])
    got = {m["id"]: m for m in helpdesk.read_messages(ids)}
    bad, context = [], []
    for i in ids:
        m = got.get(i)
        if m is None:
            bad.append(f"evidence message {i} not found")
        elif m.get("ticket_id") != case["ticket_id"]:
            bad.append(f"evidence message {i} is not on ticket {case['ticket_id']}")
        elif not m.get("date"):
            bad.append(f"evidence message {i} has no date")
        elif norm_dt(m["date"]) < cutoff:
            context.append(i)
    if bad:
        raise ReferenceDefect(f"{len(bad)} evidence problem(s): " + "; ".join(bad[:5]))
    case = dict(case)
    case["evidence_ids"] = [i for i in ids if i not in set(context)]
    case["context_evidence_ids"] = sorted(set(case.get("context_evidence_ids") or []) | set(context))
    return case


def audit_inputs(inputs: dict, target_id: Any) -> None:
    """After the barrier: nothing dated at or after the cutoff, no self-reference."""
    cutoff = inputs["cutoff"]
    bad = []
    for m in inputs["target"].get("messages", []):
        if not m.get("date") or norm_dt(m["date"]) >= cutoff:
            bad.append("target message at/after cutoff")
    for s in inputs["similar"]:
        if s["id"] == target_id:
            bad.append("target among similar tickets")
        if not s.get("created") or norm_dt(s["created"]) >= cutoff:
            bad.append(f"similar #{s['id']} created at/after cutoff")
        for m in s.get("messages", []):
            if not m.get("date") or norm_dt(m["date"]) >= cutoff:
                bad.append(f"similar #{s['id']} message at/after cutoff")
    if bad:
        raise LeakError(f"{len(bad)} leak(s): " + "; ".join(sorted(set(bad))[:5]))


def search_text(ticket: dict, messages: list) -> str:
    first = next((m["body"] for m in messages if m.get("author") == "customer"), "")
    return f"{ticket.get('subject', '')} {first}"


def build_inputs(helpdesk: Any, case: dict, *, similar_limit: int = 3,
                 barrier: bool = True) -> dict:
    """Assemble what the generator sees for one golden case, as of its cutoff.

    barrier=False exists ONLY to demonstrate the leak (example/ and tests). It
    skips the barrier and the audit, which is exactly what a naive replay does.
    """
    cutoff = norm_dt(case["cutoff"])
    if case.get("evidence_ids"):
        case = audit_evidence(helpdesk, case, cutoff)
    raw = helpdesk.get_ticket(case["ticket_id"])
    if raw is None:
        raise LookupError(f"ticket {case['ticket_id']} not found")

    if not barrier:
        target = dict(raw, messages=helpdesk.get_messages(case["ticket_id"]))
        hits = [t for t in helpdesk.search_tickets(search_text(raw, target["messages"]),
                                                   similar_limit + 1)
                if t["id"] != case["ticket_id"]][:similar_limit]
        similar = [dict(t, messages=helpdesk.get_messages(t["id"])) for t in hits]
        return {"target": target, "similar": similar, "cutoff": cutoff,
                "barrier_stats": None}

    hidden = list(case.get("evidence_ids") or []) + list(case.get("hidden_message_ids") or [])
    view = Barrier(helpdesk, cutoff=cutoff, target_id=case["ticket_id"],
                   exclude_message_ids=hidden)
    ticket = view.rewind([copy.deepcopy(raw)])[0]
    if norm_dt(ticket.get("created") or "9999") >= cutoff:
        raise LeakError(f"ticket {case['ticket_id']} was created at/after its own cutoff")
    target = dict(ticket, messages=view.get_messages(case["ticket_id"]))
    hits = view.search_tickets(search_text(ticket, target["messages"]), similar_limit)
    similar = [dict(t, messages=view.get_messages(t["id"])) for t in hits]
    inputs = {"target": target, "similar": similar, "cutoff": cutoff,
              "barrier_stats": dict(view.stats)}
    audit_inputs(inputs, case["ticket_id"])
    return inputs
