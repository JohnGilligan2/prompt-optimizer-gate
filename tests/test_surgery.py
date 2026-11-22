import pytest

import world
from promptgate import SurgeryError

L, CUR = world.LAYOUT, world.CURRENT_PROMPT


def test_round_trip_and_middle_is_byte_identical():
    _pre, mid, _rules = L.check_round_trip(CUR)
    cand = L.apply(CUR, world.PROPOSAL_B["preamble"], world.PROPOSAL_B["rules"])
    assert mid in cand and cand != CUR


def test_missing_anchor_fails_closed():
    with pytest.raises(SurgeryError, match="anchors"):
        L.split(CUR.replace("## RULES", "## THINGS"))


def test_lost_required_literal_rejected():
    rules = world.PROPOSAL_B["rules"].replace("[assistant-note]", "a header")
    with pytest.raises(SurgeryError, match="required literal"):
        L.apply(CUR, world.PROPOSAL_B["preamble"], rules)


def test_reproduced_immutable_heading_rejected():
    with pytest.raises(SurgeryError, match="immutable heading"):
        L.apply(CUR, world.PROPOSAL_B["preamble"] + "\n## CONTEXT\nmore", world.PROPOSAL_B["rules"])


def test_deleted_rule_family_rejected():
    rules = world.PROPOSAL_B["rules"].replace("- Never promise a fix time.\n", "")
    with pytest.raises(SurgeryError, match="fix time"):
        L.apply(CUR, world.PROPOSAL_B["preamble"], rules)


def test_gutted_preamble_rejected():
    with pytest.raises(SurgeryError, match="preamble shrank"):
        L.apply(CUR, "Hi.\n", world.PROPOSAL_B["rules"])


def test_sprawl_rejected():
    rules = world.PROPOSAL_B["rules"] + "- Be thorough.\n" * 40
    with pytest.raises(SurgeryError, match="x the current prompt"):
        L.apply(CUR, world.PROPOSAL_B["preamble"], rules)
