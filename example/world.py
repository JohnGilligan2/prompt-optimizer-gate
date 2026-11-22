"""A small synthetic helpdesk, golden set and stub models for the demo and tests.

Everything here is made up: the companies, the tickets, the dates. The domain
is a small managed IT provider's support desk. The stubs are deterministic
stand-ins for the three model roles, so the whole loop runs offline:

  * `generate(prompt, inputs)` plays the note generator. It follows rules it
    can find in the prompt text ("cite the closest past ticket", "name the
    likely root cause", "under N words"), which is enough to make prompt edits
    change scores in the same direction a real model's would.
  * `judge(fixture, output, rep)` plays the LLM judge: it scores the note
    against the golden case's reference root cause and fix, with a little
    deterministic noise so repeats matter.
  * `StubModel` plays the optimizer's model: it reads the corpus for the
    weakness summary and plays back scripted candidate rewrites.
"""

from __future__ import annotations

import hashlib
import re

from promptgate import PromptLayout

NOTE_HEADER = "[assistant-note]"

CURRENT_PROMPT = """You write the internal note a support engineer reads before working a ticket.
Be plain and specific. The engineer has not opened the ticket yet.

## CONTEXT
Company: a small managed IT provider. Customers are offices of 5 to 50 people.
Engineers prefer short notes they can act on in a minute.

## RULES
- Start with the line [assistant-note].
- Cite the closest past ticket and what fixed it.
- Never promise a fix time.
- No em dashes and no smart quotes.
"""

LAYOUT = PromptLayout(
    immutable_anchor="## CONTEXT",
    rules_anchor="## RULES",
    required_literals=(NOTE_HEADER,),
    required_fragments_ci=("fix time", "em dash"),
)

# Candidate A adds the root-cause rule the critiques ask for, plus a length cap.
# The cap helps nothing on short tickets and wrecks the one long ticket.
PROPOSAL_A = {
    "preamble": "You write the internal note a support engineer reads before working a ticket.\n"
                "Be plain and specific. The engineer has not opened the ticket yet.\n",
    "rules": "## RULES\n- Start with the line [assistant-note].\n"
             "- Name the likely root cause, from the closest past ticket.\n"
             "- Cite the closest past ticket and what fixed it.\n"
             "- Keep the note under 40 words.\n"
             "- Never promise a fix time.\n- No em dashes and no smart quotes.\n",
    "change_notes": "added 'name the likely root cause' (weakness 1); added a 40 word cap "
                    "(engineers prefer short notes)",
}

# Candidate B is the same root-cause rule without the cap.
PROPOSAL_B = {
    "preamble": PROPOSAL_A["preamble"],
    "rules": "## RULES\n- Start with the line [assistant-note].\n"
             "- Name the likely root cause, from the closest past ticket.\n"
             "- Cite the closest past ticket and what fixed it.\n"
             "- Never promise a fix time.\n- No em dashes and no smart quotes.\n",
    "change_notes": "added 'name the likely root cause' (weakness 1)",
}


# -- the helpdesk ---------------------------------------------------------------

def _t(i, subject, customer, created, stage="Solved", assignee="tier1"):
    return {"id": i, "subject": subject, "customer": customer, "created": created,
            "stage": stage, "assignee": assignee, "updated": "2026-04-01T08:00:00"}


def _m(i, ticket_id, date, author, body):
    return {"id": i, "ticket_id": ticket_id, "date": date, "author": author, "body": body}


TICKETS = [
    # history, solved well before any golden cutoff
    _t(101, "Outbound email stuck in outbox", "Maple Row Accounting", "2026-01-05T09:10:00"),
    _t(102, "Printer shows offline after power cut", "Bluefin Print Shop", "2026-01-12T14:00:00"),
    _t(103, "VPN connects but no file shares", "Copperleaf Studio", "2026-01-20T08:30:00"),
    _t(104, "Nightly backup job failed", "Tidewater Bikes", "2026-02-02T07:45:00"),
    _t(105, "Wi-Fi drops in back office", "Maple Row Accounting", "2026-02-10T11:00:00"),
    # golden targets: stage reads Solved today, it read New at the cutoff
    _t(201, "Email stuck in outbox again", "Bluefin Print Shop", "2026-03-02T09:00:00"),
    _t(202, "Printer offline on second floor", "Maple Row Accounting", "2026-03-03T10:00:00"),
    _t(203, "VPN up but cannot open shared drive", "Tidewater Bikes", "2026-03-04T08:00:00"),
    _t(204, "Backup job failed overnight with a long error about the destination volume "
            "running out of free space on the external disk attached to the office "
            "server, and the retry at six this morning failed the same way so nobody "
            "has a good copy from last night", "Copperleaf Studio", "2026-03-05T07:30:00"),
    _t(205, "Wi-Fi keeps dropping in conference room", "Tidewater Bikes", "2026-03-06T13:00:00"),
    _t(206, "Label printer prints blank labels", "Bluefin Print Shop", "2026-03-07T10:00:00"),
    # from the future: repeats of 201 and 206, created after their cutoffs
    _t(301, "Email stuck in outbox, same as last week", "Bluefin Print Shop", "2026-03-09T09:00:00"),
    _t(302, "Label printer blank labels again", "Bluefin Print Shop", "2026-03-12T09:00:00"),
]

MESSAGES = [
    _m(1001, 101, "2026-01-05T09:10:00", "customer", "Emails sit in the outbox and never send."),
    _m(1002, 101, "2026-01-05T10:20:00", "engineer",
       "Cause: SMTP relay password expired. Fix: reset the relay credential and resent the queue."),
    _m(1003, 102, "2026-01-12T14:00:00", "customer", "Printer says offline since the power cut."),
    _m(1004, 102, "2026-01-12T15:00:00", "engineer",
       "Cause: printer picked up a new DHCP address. Fix: set a DHCP reservation for the printer."),
    _m(1005, 103, "2026-01-20T08:30:00", "customer", "VPN connects but I cannot see the file shares."),
    _m(1006, 103, "2026-01-20T09:40:00", "engineer",
       "Cause: split tunnel route missing for the file server subnet. Fix: added the route to the VPN profile."),
    _m(1007, 104, "2026-02-02T07:45:00", "customer", "The nightly backup job failed."),
    _m(1008, 104, "2026-02-02T09:00:00", "engineer",
       "Cause: backup disk full. Fix: pruned old restore points and set a free space alert."),
    _m(1009, 105, "2026-02-10T11:00:00", "customer", "Wi-Fi drops every few minutes in the back office."),
    _m(1010, 105, "2026-02-10T12:30:00", "engineer",
       "Cause: channel overlap with a neighbor network. Fix: moved the access point to channel 11."),

    _m(2001, 201, "2026-03-02T09:00:00", "customer", "Email is stuck in the outbox again, nothing sends."),
    _m(2002, 201, "2026-03-02T11:00:00", "engineer",
       "Cause: SMTP relay password expired again. Fix: reset the relay credential and set a reminder."),
    _m(2003, 202, "2026-03-03T10:00:00", "customer", "The second floor printer shows offline."),
    _m(2004, 202, "2026-03-03T12:00:00", "engineer",
       "Cause: new DHCP address after a router swap. Fix: set a DHCP reservation."),
    _m(2005, 203, "2026-03-04T08:00:00", "customer", "VPN is up but the shared drive will not open."),
    _m(2006, 203, "2026-03-04T09:30:00", "engineer",
       "Cause: split tunnel route missing. Fix: added the route to the VPN profile."),
    _m(2007, 204, "2026-03-05T07:30:00", "customer", "Backup failed overnight, error mentions free space."),
    _m(2008, 204, "2026-03-05T09:15:00", "engineer",
       "Cause: backup disk full. Fix: pruned old restore points."),
    _m(2009, 205, "2026-03-06T13:00:00", "customer", "Wi-Fi keeps dropping in the conference room."),
    _m(2010, 205, "2026-03-06T15:00:00", "engineer",
       "Cause: channel overlap with a neighbor network. Fix: moved the access point to channel 11."),

    _m(2011, 206, "2026-03-07T10:00:00", "customer", "The label printer feeds but every label comes out blank."),
    _m(2012, 206, "2026-03-07T11:30:00", "engineer",
       "Cause: thermal head worn out. Fix: replaced the print head."),

    _m(3001, 301, "2026-03-09T09:00:00", "customer", "Email stuck in outbox again."),
    _m(3002, 301, "2026-03-09T09:20:00", "engineer",
       "Cause: SMTP relay password expired. Fix: reset the relay credential, moved it to a service account."),
    _m(3003, 302, "2026-03-12T09:00:00", "customer", "Label printer is printing blank labels again."),
    _m(3004, 302, "2026-03-12T09:40:00", "engineer",
       "Cause: thermal head worn out on the spare too. Fix: replaced the print head."),
]

# (date, old, new) changes for tracked fields
TRACKING = {
    i: {"stage": [(f"{d}T{h}", "New", "In Progress"), (f"{d}T{h2}", "In Progress", "Solved")]}
    for i, d, h, h2 in [
        (201, "2026-03-02", "10:00:00", "11:05:00"),
        (202, "2026-03-03", "11:00:00", "12:05:00"),
        (203, "2026-03-04", "09:00:00", "09:35:00"),
        (204, "2026-03-05", "08:30:00", "09:20:00"),
        (205, "2026-03-06", "14:00:00", "15:05:00"),
        (206, "2026-03-07", "11:00:00", "11:35:00"),
    ]
}

_WORD = re.compile(r"[a-z0-9-]+")
_STOP = {"the", "a", "an", "in", "on", "of", "to", "and", "but", "is", "i", "it", "not",
         "with", "after", "since", "will", "every", "few", "about", "nothing"}


def _words(text):
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


class Helpdesk:
    """In-memory helpdesk with the read methods pit.Barrier expects."""

    def __init__(self, tickets=TICKETS, messages=MESSAGES, tracking=TRACKING):
        self.tickets = {t["id"]: dict(t) for t in tickets}
        self.messages = [dict(m) for m in messages]
        self.tracking = tracking

    def search_tickets(self, text, limit):
        q = _words(text)
        scored = []
        for t in self.tickets.values():
            body = " ".join(m["body"] for m in self.messages if m["ticket_id"] == t["id"]
                            and m["author"] == "customer")
            overlap = len(q & _words(t["subject"] + " " + body))
            if overlap:
                scored.append((-overlap, t["id"]))
        return [dict(self.tickets[i]) for _, i in sorted(scored)[:limit]]

    def get_ticket(self, ticket_id):
        t = self.tickets.get(ticket_id)
        return dict(t) if t else None

    def get_messages(self, ticket_id):
        return [dict(m) for m in self.messages if m["ticket_id"] == ticket_id]

    def read_messages(self, ids):
        ids = set(ids)
        return [dict(m) for m in self.messages if m["id"] in ids]

    def tracking_changes(self, ids):
        return {i: self.tracking[i] for i in ids if i in self.tracking}

    def post_message(self, ticket_id, body):  # a write, so the barrier must refuse it
        raise AssertionError("a replay reached the real helpdesk's write path")


# -- the golden set -----------------------------------------------------------
# cutoff: the moment before an engineer worked the ticket.
# evidence_ids: the messages the reference answer was taken from.
# reference: what the note should get right, as keywords the stub judge looks for.

GOLDEN = [
    {"case_id": "case-201", "ticket_id": 201, "cutoff": "2026-03-02T09:30:00",
     "evidence_ids": [2002], "reference": {"cause": "relay password", "fix": "reset the relay credential"}},
    {"case_id": "case-202", "ticket_id": 202, "cutoff": "2026-03-03T10:30:00",
     "evidence_ids": [2004], "reference": {"cause": "dhcp address", "fix": "dhcp reservation"}},
    {"case_id": "case-203", "ticket_id": 203, "cutoff": "2026-03-04T08:30:00",
     "evidence_ids": [2006], "reference": {"cause": "split tunnel route", "fix": "added the route"}},
    {"case_id": "case-204", "ticket_id": 204, "cutoff": "2026-03-05T08:00:00",
     "evidence_ids": [2008], "reference": {"cause": "disk full", "fix": "pruned old restore points"}},
    {"case_id": "case-205", "ticket_id": 205, "cutoff": "2026-03-06T13:30:00",
     "evidence_ids": [2010], "reference": {"cause": "channel overlap", "fix": "channel 11"}},
    # nothing like this in the history before the cutoff: the honest score is low
    {"case_id": "case-206", "ticket_id": 206, "cutoff": "2026-03-07T10:30:00",
     "evidence_ids": [2012], "reference": {"cause": "thermal head", "fix": "replaced the print head"}},
]


# -- production cases the optimizer learns from ---------------------------------
# judge score + comment, and a human grade on some (for the kappa cross-check).

CASES = [
    {"id": "p-01", "timestamp": "2026-04-06T09:00:00", "source": "production", "score": 3,
     "human": 3, "comment": "Cites the right past fix but never says what caused it.",
     "output": "[assistant-note] Ticket: Mailbox full warning. Closest past ticket #88. What fixed it then: archived old mail."},
    {"id": "p-02", "timestamp": "2026-04-06T11:00:00", "source": "production", "score": 3,
     "human": 3, "comment": "Useful fix, no root cause, engineer has to work out why.",
     "output": "[assistant-note] Ticket: Scanner not saving to folder. Closest past ticket #91. What fixed it then: re-mapped the share."},
    {"id": "p-03", "timestamp": "2026-04-07T08:30:00", "source": "production", "score": 4,
     "human": 4, "comment": "Clear and specific.",
     "output": "[assistant-note] Ticket: New starter laptop. Closest past ticket #77. What fixed it then: ran the standard build."},
    {"id": "p-04", "timestamp": "2026-04-07T10:00:00", "source": "production", "score": 2,
     "human": 3, "comment": "No root cause named; the fix may not apply.",
     "output": "[assistant-note] Ticket: Calendar not syncing on phone. Closest past ticket #63. What fixed it then: removed and re-added the account."},
    {"id": "p-05", "timestamp": "2026-04-08T09:15:00", "source": "production", "score": 5,
     "human": 5, "comment": "Names the cause and the fix.",
     "output": "[assistant-note] Ticket: Shared drive read-only. Likely cause: group membership removed. Closest past ticket #70. What fixed it then: re-added the group."},
    {"id": "p-06", "timestamp": "2026-04-08T14:00:00", "source": "production", "score": 3,
     "human": 2, "comment": "Fix cited, root cause missing.",
     "output": "[assistant-note] Ticket: Slow logins. Closest past ticket #59. What fixed it then: cleared the profile cache."},
    {"id": "p-07", "timestamp": "2026-04-09T09:00:00", "source": "production", "score": 4,
     "human": 4, "comment": "Good, short.",
     "output": "[assistant-note] Ticket: Password reset. Closest past ticket #12. What fixed it then: self-service reset link."},
    # never learned from: a gate replay, and a synthetic test invocation
    {"id": "g-01", "timestamp": "2026-04-09T10:00:00", "source": "gate", "score": 1,
     "comment": "gate replay", "output": "(gate)"},
    {"id": "t-01", "timestamp": "2026-04-09T10:05:00", "source": "production", "synthetic": True,
     "score": 1, "comment": "test invocation", "output": "(test)"},
]


# -- the stubs --------------------------------------------------------------------

_CAUSE = re.compile(r"Cause:\s*([^.]+)\.")
_FIX = re.compile(r"Fix:\s*([^.]+)\.")
_LIMIT = re.compile(r"under (\d+) words")


def _cause_fix(messages):
    cause = fix = None
    for m in messages:
        if m.get("author") != "engineer":
            continue
        c, f = _CAUSE.search(m["body"]), _FIX.search(m["body"])
        cause = cause or (c.group(1) if c else None)
        fix = fix or (f.group(1) if f else None)
    return cause, fix


def generate(prompt, inputs):
    """Stub note generator. Deterministic; follows the rules it finds in the prompt."""
    p = prompt.lower()
    target = inputs["target"]
    own_cause, own_fix = _cause_fix(target.get("messages", []))   # only visible if leaked
    best = inputs["similar"][0] if inputs["similar"] else None
    sim_cause, sim_fix = _cause_fix(best["messages"]) if best else (None, None)
    cause, fix = own_cause or sim_cause, own_fix or sim_fix

    parts = []
    if NOTE_HEADER in prompt:
        parts.append(NOTE_HEADER)
    parts.append(f"Ticket: {target['subject']}.")
    if "name the likely root cause" in p and cause:
        parts.append(f"Likely cause: {cause}.")
    if "cite the closest past ticket" in p and best:
        parts.append(f"Closest past ticket #{best['id']}. What fixed it then: {fix or 'not recorded'}.")
    text = " ".join(parts)
    cap = _LIMIT.search(p)
    if cap:
        text = " ".join(text.split()[: int(cap.group(1))])
    return text


def judge(fixture, output, rep):
    """Stub judge: 1 point, plus 2 for the right root cause, plus 2 for the right fix.

    A little deterministic noise (about one replay in seven loses a point), so a
    single replay can mislead and repeats are worth paying for.
    """
    ref, low = fixture["reference"], (output or "").lower()
    cause_ok, fix_ok = ref["cause"] in low, ref["fix"] in low
    score = 1 + 2 * cause_ok + 2 * fix_ok
    h = int(hashlib.sha256(f"{fixture['id']}|{rep}|{output}".encode()).hexdigest()[:8], 16)
    if h % 7 == 0 and score > 1:
        score -= 1
    missing = [n for n, ok in (("root cause", cause_ok), ("fix", fix_ok)) if not ok]
    return {"score": score, "comment": "names cause and fix" if not missing
            else "missing " + " and ".join(missing)}


class StubModel:
    """Stub optimizer model. Weakness summary from the corpus; scripted rewrites."""

    def __init__(self, proposals):
        self.proposals = list(proposals)
        self.calls = []

    def __call__(self, system, user, schema):
        self.calls.append(system.splitlines()[0])
        if "weaknesses" in schema["properties"]:
            ids = sorted(set(re.findall(r"case (p-\d+) .*\n\s+judge: \"[^\"]*(?:root cause|caused)",
                                        user)))
            if len(ids) < 2:
                return {"weaknesses": [], "summary": "No recurring weakness."}
            return {"weaknesses": [{
                "theme": "Notes cite a past fix but never name the likely root cause",
                "evidence": f"cases {', '.join(ids)}: judge says the root cause is missing",
                "suggested_rule_change": "Add: name the likely root cause, from the closest past ticket.",
            }], "summary": "Fixes are cited well; root cause is the gap."}
        return self.proposals.pop(0)


def fixtures(helpdesk=None, barrier=True):
    """Golden cases as gate fixtures, inputs built point-in-time (or naively)."""
    from promptgate import build_inputs
    hd = helpdesk or Helpdesk()
    return [{"id": c["case_id"], "reference": c["reference"],
             "inputs": build_inputs(hd, c, barrier=barrier)} for c in GOLDEN]
