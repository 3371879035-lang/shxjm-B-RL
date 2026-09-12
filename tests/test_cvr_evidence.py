import dataclasses

import pytest

from brl.cvr.evidence import CoverageEvidenceLedger, CoverageNode, FutureCoveragePlan
from brl.cvr.ids import ChannelId, PlanNodeId, StationId


def test_only_accepted_no_signal_enters_negative_evidence():
    ledger = CoverageEvidenceLedger()
    channel = ChannelId(3)
    assert not ledger.accept(channel, (10.0, 20.0), "direction", "r1", accepted=True)
    assert not ledger.accept(channel, (10.0, 20.0), "no_signal", "r2", accepted=False)
    assert ledger.accept(channel, (10.0, 20.0), "no_signal", "r3", accepted=True)
    assert ledger.negative_points(channel) == ((10.0, 20.0),)
    assert not ledger.accept(channel, (10.0, 20.0), "no_signal", "r3", accepted=True)


def test_plan_mutation_increments_version_and_removes_old_witness():
    channel = ChannelId(1)
    old = CoverageNode(
        PlanNodeId("fixed-1"), (1.0, 2.0), frozenset({channel}), StationId(1)
    )
    new = CoverageNode(PlanNodeId("variable-a"), (3.0, 4.0), frozenset({channel}), None)
    plan = FutureCoveragePlan.initial((old,))
    changed = plan.replace((old.node_id,), (new,))
    assert changed.version == plan.version + 1
    assert changed.nodes == (new,)
    with pytest.raises(dataclasses.FrozenInstanceError):
        changed.version = 7


def test_completing_a_channel_removes_empty_node_and_versions_plan():
    channel = ChannelId(1)
    node = CoverageNode(PlanNodeId("fixed-0"), (0.0, 0.0), frozenset({channel}), StationId(0))
    plan = FutureCoveragePlan.initial((node,))
    complete = plan.complete_node_channel(node.node_id, channel)
    assert complete.version == 2
    assert complete.nodes == ()
