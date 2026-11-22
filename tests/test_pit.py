import pytest

import world
from promptgate import Barrier, LeakError, ReadOnlyViolation, build_inputs
from promptgate.pit import ReferenceDefect, audit_inputs, value_as_of


def case(case_id):
    return next(c for c in world.GOLDEN if c["case_id"] == case_id)


def test_value_as_of():
    ch = [("2026-03-02T10:00:00", "New", "In Progress"),
          ("2026-03-02T11:00:00", "In Progress", "Solved")]
    assert value_as_of(ch, "2026-03-02T09:00:00", "Solved") == "New"
    assert value_as_of(ch, "2026-03-02T10:30:00", "Solved") == "In Progress"
    assert value_as_of(ch, "2026-03-03T00:00:00", "Solved") == "Solved"
    assert value_as_of([], "2026-03-02T09:00:00", "Solved") == "Solved"


def test_point_in_time_inputs_hide_the_future():
    inputs = build_inputs(world.Helpdesk(), case("case-201"))
    t = inputs["target"]
    assert t["stage"] == "New"                         # rewound from Solved
    assert t["updated"] is None
    assert [m["id"] for m in t["messages"]] == [2001]  # the fix (2002) is hidden
    ids = [s["id"] for s in inputs["similar"]]
    assert 201 not in ids and 301 not in ids           # no self, nothing from the future
    assert 101 in ids                                  # the honest neighbor survives


def test_naive_replay_leaks_the_answer():
    inputs = build_inputs(world.Helpdesk(), case("case-206"), barrier=False)
    bodies = " ".join(m["body"] for m in inputs["target"]["messages"])
    assert "replaced the print head" in bodies
    assert 302 in [s["id"] for s in inputs["similar"]]
    with pytest.raises(LeakError):
        audit_inputs(inputs, 206)


def test_leak_inflates_the_score():
    cur = world.CURRENT_PROMPT
    pit = {f["id"]: f for f in world.fixtures()}
    naive = {f["id"]: f for f in world.fixtures(barrier=False)}
    s_pit = world.judge(pit["case-206"], world.generate(cur, pit["case-206"]["inputs"]), 1)["score"]
    s_naive = world.judge(naive["case-206"], world.generate(cur, naive["case-206"]["inputs"]), 1)["score"]
    assert s_naive > s_pit


def test_barrier_refuses_writes():
    view = Barrier(world.Helpdesk(), cutoff="2026-03-02T09:30:00", target_id=201)
    with pytest.raises(ReadOnlyViolation):
        view.post_message(201, "hello")


def test_barrier_drops_late_messages_on_older_tickets():
    view = Barrier(world.Helpdesk(), cutoff="2026-01-05T10:00:00", target_id=999)
    assert [m["id"] for m in view.get_messages(101)] == [1001]


def test_evidence_dated_before_cutoff_becomes_context():
    c = dict(case("case-201"), evidence_ids=[2001, 2002])
    inputs = build_inputs(world.Helpdesk(), c)
    assert [m["id"] for m in inputs["target"]["messages"]] == [2001]


def test_foreign_evidence_is_a_reference_defect():
    c = dict(case("case-201"), evidence_ids=[1002])     # a message on another ticket
    with pytest.raises(ReferenceDefect):
        build_inputs(world.Helpdesk(), c)


def test_ticket_created_after_its_own_cutoff_is_a_leak():
    c = dict(case("case-201"), cutoff="2026-03-01T00:00:00", evidence_ids=[])
    with pytest.raises(LeakError):
        build_inputs(world.Helpdesk(), c)
