from brl.coverage import s3_points
from brl.cvr.evidence import CoverageNode, FutureCoveragePlan
from brl.cvr.ids import ChannelId, PlanNodeId, StationId
from brl.cvr.waypoints import PublicCVRSnapshot, PublicCVRTrack, WaypointGenerator


def _q3_plan():
    channels = frozenset(ChannelId(channel) for channel in range(1, 21))
    return FutureCoveragePlan.initial(
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


def test_same_public_snapshot_produces_identical_candidates():
    snapshot = PublicCVRSnapshot(3, (0.0, 0.0), 1, (), _q3_plan())
    first = WaypointGenerator().generate(snapshot)
    second = WaypointGenerator().generate(snapshot)
    assert first == second


def test_replacement_never_reuses_a_legacy_station_id():
    snapshot = PublicCVRSnapshot(
        3,
        (0.0, 0.0),
        1,
        (PublicCVRTrack(ChannelId(1), (800.0, 400.0), 70.0, "probe", (900.0, 300.0)),),
        _q3_plan(),
    )
    proposals = WaypointGenerator().replacement_proposals(snapshot)
    assert proposals
    assert all(
        node.legacy_station is None for proposal in proposals for node in proposal.added
    )


def test_generated_points_stay_inside_target_disk():
    snapshot = PublicCVRSnapshot(
        3,
        (1700.0, 0.0),
        1,
        (PublicCVRTrack(ChannelId(1), (1800.0, 1800.0), 500.0, "probe", (2500.0, 2500.0)),),
        _q3_plan(),
    )
    assert all(x * x + y * y <= 1800.0**2 + 1e-4 for x, y in WaypointGenerator().generate(snapshot))
