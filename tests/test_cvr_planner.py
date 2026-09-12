import pytest

from brl.coverage import s3_points
from brl.cvr.certify import CoverageCertificateEngine
from brl.cvr.evidence import CoverageEvidenceLedger, CoverageNode, FutureCoveragePlan
from brl.cvr.ids import ChannelId, PlanNodeId, StationId
from brl.cvr.planner import MacroTask, SegmentPlanner, SegmentPlannerConfig
from brl.cvr.waypoints import PublicCVRSnapshot, WaypointGenerator


@pytest.fixture
def q3_snapshot():
    channels = frozenset(ChannelId(channel) for channel in range(1, 21))
    plan = FutureCoveragePlan.initial(
        tuple(
            CoverageNode(
                PlanNodeId(f"fixed-{index}"),
                tuple(map(float, point)),
                channels,
                StationId(index),
            )
            for index, point in enumerate(s3_points())
        )
    )
    return PublicCVRSnapshot(3, (0.0, 0.0), 1, (), plan)


def test_planner_never_returns_an_uncertified_replacement(q3_snapshot):
    planner = SegmentPlanner(CoverageCertificateEngine(3), WaypointGenerator())
    chosen = planner.plan(q3_snapshot, CoverageEvidenceLedger(), exit_position=(1200.0, 0.0))
    assert chosen.certificate.covered
    assert all(result.covered for _, result in chosen.channel_certificates)
    assert all(mutation.plan_after.version > q3_snapshot.plan.version for mutation in chosen.mutations)


def test_segment_cost_includes_return_to_declared_exit(q3_snapshot):
    planner = SegmentPlanner(CoverageCertificateEngine(3), WaypointGenerator())
    task = MacroTask("measure", "n", (100.0, 0.0), (1, 2))
    near_exit = planner.segment_cost(q3_snapshot, (task,), (200.0, 0.0))
    far_exit = planner.segment_cost(q3_snapshot, (task,), (-1800.0, 0.0))
    assert far_exit > near_exit


def test_zero_planning_budget_returns_frozen_baseline_segment(q3_snapshot):
    config = SegmentPlannerConfig(max_call_s=0.0, max_total_s=0.0)
    planner = SegmentPlanner(CoverageCertificateEngine(3), WaypointGenerator(), config)
    segment = planner.plan(q3_snapshot, CoverageEvidenceLedger(), exit_position=(1200.0, 0.0))
    assert segment.reason == "planning_budget_fallback"
    assert not segment.mutations


def test_fixed_segment_mode_permits_reordering_without_mutating_plan(q3_snapshot):
    planner = SegmentPlanner(CoverageCertificateEngine(3), WaypointGenerator())
    segment = planner.plan(
        q3_snapshot, CoverageEvidenceLedger(), exit_position=(0.0, 0.0), allow_replacements=False
    )
    assert segment.reason == "frozen_baseline"
    assert not segment.mutations
