# Certified Variable Waypoint Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and locally validate a deterministic CVR candidate that may replace one or two pending S3/S25 stations with certified variable measurement points and plans 2–4 macro-task route segments without weakening source-clearance or channel-absence evidence.

**Architecture:** Keep ISR-v1 as the unchanged fallback. Add a typed, immutable CVR plan and per-channel negative-observation ledger; verify Q3 disk coverage and Q4 triangulation coverage before accepting any future-plan mutation. Reuse JSP's resumable bilateral localization execution, but give CVR its own task identifiers, segment planner, completion certificate, incremental logs, and paired experiment runner.

**Tech Stack:** Python 3, NumPy, SciPy spatial geometry, pytest, CSV/JSONL experiment artifacts, existing `RadioEnv`, `CheckedView`, `BilateralState`, S3/S25, and ISR-v1.

---

## File map

Create these focused modules:

- `brl/cvr/ids.py`: runtime-distinct station, channel, and plan-node identifiers.
- `brl/cvr/evidence.py`: immutable negative observations, future measurement nodes, versioned plans, and evidence ledger.
- `brl/cvr/certify.py`: Q3 exact candidate-set coverage radius and Q4 sufficient triangulation certificate.
- `brl/cvr/waypoints.py`: deterministic variable-point generation and `replace-1`/`replace-2` mutations.
- `brl/cvr/planner.py`: typed macro tasks, exact segment cost, bounded beam search, and replacement selection.
- `brl/cvr/policy.py`: policy execution, feedback commit, custom completion certificate, logging, and ISR fallback.
- `brl/cvr/__init__.py`: public CVR exports.
- `scripts/run_cvr_experiments.py`: resumable local ISR/SEG-FIXED/CVR paired experiments and gates.
- `scripts/audit_cvr_certificates.py`: offline replay of evidence and plan-version transitions.
- `scripts/analyze_cvr_local.py`: cost and mechanism summary.
- `docs/CVR_RUNBOOK.md`: exact local commands and evidence interpretation.

Modify only these existing files:

- `brl/jsp/policy.py`: correct typed meaning in remaining-station scoring.
- `brl/jsp/snapshot.py`: prevent scan candidates from reading same-numbered channel tracks.
- `tests/test_jsp_policy.py`: lock both bug fixes.

Add tests:

- `tests/test_cvr_ids.py`
- `tests/test_cvr_evidence.py`
- `tests/test_cvr_certify.py`
- `tests/test_cvr_waypoints.py`
- `tests/test_cvr_planner.py`
- `tests/test_cvr_policy.py`
- `tests/test_cvr_audit.py`

Do not modify `brl/local_env.py`, `brl/remote.py`, ISR-v1, the paper, or official batch automation in this implementation. CVR's absence certificate remains policy-owned and independently audited; original stations measured by CVR still use their real `coverage_idx`, while variable points use `coverage_idx=None`.

### Task 1: Correct the frozen JSP identifier collisions

**Files:**
- Modify: `brl/jsp/policy.py:246-285`
- Modify: `brl/jsp/snapshot.py:58-85`
- Modify: `tests/test_jsp_policy.py`

- [ ] **Step 1: Write failing regression tests**

Append tests which make a locate candidate use the same integer as a pending station, and make a scan candidate use the same integer as a discovered channel:

```python
from brl.jsp.policy import CandidateAction


def test_localize_candidate_does_not_remove_same_numbered_station_from_cost():
    env = RadioEnv(mode=4, n_sources=1, seed=606, bearing_decimals=2)
    env.reset(seed=606, sources=[Source(1, np.array([1400.0, 0.0]), 1500.0)])
    env.error_field.value = lambda *args: 0.0
    policy = JSPPolicy(env)
    policy._scan(0, 1)
    stations = [1, 2]
    candidate = CandidateAction("locate1", 1, tuple(policy._locator_action(1)[1]))
    score_with_both = policy._analytic(candidate, stations)
    score_without_one = policy._analytic(candidate, [2])
    assert score_with_both > score_without_one


def test_scan_candidate_does_not_read_same_numbered_channel_track():
    env = RadioEnv(mode=3, n_sources=1, seed=607, bearing_decimals=2)
    env.reset(seed=607, sources=[Source(1, np.array([900.0, 0.0]), 1500.0)])
    env.error_field.value = lambda *args: 0.0
    policy = JSPPolicy(env)
    policy._scan(0, 1)
    scan = CandidateAction("scanall", 1, tuple(policy.points[1]), 20)
    public = policy._public_candidate(scan)
    row = candidate_features(policy.public_snapshot(), public)
    radius_index = 17
    observation_index = 18
    assert row[radius_index] == 0.0
    assert row[observation_index] == 0.0
```

- [ ] **Step 2: Run the two tests and verify the known failures**

Run:

```powershell
pytest tests/test_jsp_policy.py -k "same_numbered" -v
```

Expected: both new tests fail on the frozen JSP implementation.

- [ ] **Step 3: Make scoring branch on candidate kind**

In `JSPPolicy._analytic`, replace the untyped deletion with:

```python
completed_station = candidate.key if candidate.kind in {"scan4", "scanall"} else None
remaining_points = [
    self.points[i] for i in scan_indices if i != completed_station
]
```

In `candidate_features`, replace the untyped track lookup with:

```python
source_action = candidate.kind.startswith("locate") or candidate.kind == "resolve"
track = (
    next((t for t in snapshot.tracks if t.channel == candidate.key), None)
    if source_action else None
)
```

- [ ] **Step 4: Run focused and complete JSP tests**

Run:

```powershell
pytest tests/test_jsp_policy.py tests/test_jsp_model.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the isolated correction**

```powershell
git add -- brl/jsp/policy.py brl/jsp/snapshot.py tests/test_jsp_policy.py
git commit -m "fix: separate JSP station and channel identifiers"
```

### Task 2: Add runtime-distinct CVR identifiers

**Files:**
- Create: `brl/cvr/ids.py`
- Create: `brl/cvr/__init__.py`
- Create: `tests/test_cvr_ids.py`

- [ ] **Step 1: Write the failing type-boundary tests**

```python
import pytest

from brl.cvr.ids import ChannelId, PlanNodeId, StationId, require_channel, require_station


def test_ids_with_same_integer_are_not_equal():
    assert ChannelId(1) != StationId(1)
    assert PlanNodeId("fixed-1") != PlanNodeId("fixed-2")


def test_runtime_guards_reject_the_wrong_identifier_kind():
    with pytest.raises(TypeError, match="StationId"):
        require_station(ChannelId(1))
    with pytest.raises(TypeError, match="ChannelId"):
        require_channel(StationId(1))
```

- [ ] **Step 2: Run the test and verify the import failure**

Run: `pytest tests/test_cvr_ids.py -v`

Expected: FAIL because `brl.cvr.ids` does not exist.

- [ ] **Step 3: Implement small frozen value types and guards**

```python
# brl/cvr/ids.py
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class ChannelId:
    value: int
    def __post_init__(self):
        if not 1 <= self.value <= 20:
            raise ValueError("channel must be in 1..20")


@dataclass(frozen=True, order=True)
class StationId:
    value: int
    def __post_init__(self):
        if self.value < 0:
            raise ValueError("station index must be non-negative")


@dataclass(frozen=True, order=True)
class PlanNodeId:
    value: str
    def __post_init__(self):
        if not self.value:
            raise ValueError("plan node id must be non-empty")


def require_channel(value: ChannelId) -> ChannelId:
    if not isinstance(value, ChannelId):
        raise TypeError("expected ChannelId")
    return value


def require_station(value: StationId) -> StationId:
    if not isinstance(value, StationId):
        raise TypeError("expected StationId")
    return value
```

Export the three ID classes from `brl/cvr/__init__.py`.

- [ ] **Step 4: Run the type tests**

Run: `pytest tests/test_cvr_ids.py -q`

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```powershell
git add -- brl/cvr/ids.py brl/cvr/__init__.py tests/test_cvr_ids.py
git commit -m "feat: add typed CVR task identifiers"
```

### Task 3: Build the evidence ledger and versioned future plan

**Files:**
- Create: `brl/cvr/evidence.py`
- Create: `tests/test_cvr_evidence.py`
- Modify: `brl/cvr/__init__.py`

- [ ] **Step 1: Write failing immutability and invalidation tests**

```python
import dataclasses
import pytest

from brl.cvr.evidence import CoverageNode, CoverageEvidenceLedger, FutureCoveragePlan
from brl.cvr.ids import ChannelId, PlanNodeId, StationId


def test_only_accepted_no_signal_enters_negative_evidence():
    ledger = CoverageEvidenceLedger()
    ch = ChannelId(3)
    assert not ledger.accept(ch, (10.0, 20.0), "direction", "r1", accepted=True)
    assert not ledger.accept(ch, (10.0, 20.0), "no_signal", "r2", accepted=False)
    assert ledger.accept(ch, (10.0, 20.0), "no_signal", "r3", accepted=True)
    assert ledger.negative_points(ch) == ((10.0, 20.0),)
    assert not ledger.accept(ch, (10.0, 20.0), "no_signal", "r3", accepted=True)


def test_plan_mutation_increments_version_and_removes_old_witness():
    ch = ChannelId(1)
    old = CoverageNode(PlanNodeId("fixed-1"), (1.0, 2.0), frozenset({ch}), StationId(1))
    new = CoverageNode(PlanNodeId("variable-a"), (3.0, 4.0), frozenset({ch}), None)
    plan = FutureCoveragePlan.initial((old,))
    changed = plan.replace((old.node_id,), (new,))
    assert changed.version == plan.version + 1
    assert changed.nodes == (new,)
    with pytest.raises(dataclasses.FrozenInstanceError):
        changed.version = 7
```

- [ ] **Step 2: Run tests and verify missing symbols**

Run: `pytest tests/test_cvr_evidence.py -v`

Expected: FAIL because the evidence module is absent.

- [ ] **Step 3: Implement immutable nodes/plans and a request-id ledger**

```python
# brl/cvr/evidence.py
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Iterable

from .ids import ChannelId, PlanNodeId, StationId

Point = tuple[float, float]


@dataclass(frozen=True)
class NegativeObservation:
    channel: ChannelId
    position: Point
    request_id: str


@dataclass(frozen=True)
class CoverageNode:
    node_id: PlanNodeId
    position: Point
    channels: frozenset[ChannelId]
    legacy_station: StationId | None

    def without(self, channel: ChannelId) -> "CoverageNode":
        return replace(self, channels=frozenset(c for c in self.channels if c != channel))


@dataclass(frozen=True)
class FutureCoveragePlan:
    version: int
    nodes: tuple[CoverageNode, ...]

    @classmethod
    def initial(cls, nodes: Iterable[CoverageNode]) -> "FutureCoveragePlan":
        return cls(1, tuple(nodes))

    def replace(self, removed: tuple[PlanNodeId, ...],
                added: tuple[CoverageNode, ...]) -> "FutureCoveragePlan":
        remove_set = frozenset(removed)
        existing = frozenset(n.node_id for n in self.nodes)
        if not remove_set or not remove_set.issubset(existing):
            raise ValueError("replacement must name only pending plan nodes")
        kept = tuple(n for n in self.nodes if n.node_id not in remove_set)
        ids = [n.node_id for n in kept + added]
        if len(ids) != len(set(ids)):
            raise ValueError("plan node ids must be unique")
        return FutureCoveragePlan(self.version + 1, kept + added)


class CoverageEvidenceLedger:
    def __init__(self):
        self._negative: dict[ChannelId, list[NegativeObservation]] = {}
        self._request_ids: set[str] = set()

    def accept(self, channel: ChannelId, position: Point, result: str,
               request_id: str, *, accepted: bool) -> bool:
        if not accepted or result != "no_signal" or request_id in self._request_ids:
            return False
        self._request_ids.add(request_id)
        obs = NegativeObservation(channel, tuple(map(float, position)), request_id)
        self._negative.setdefault(channel, []).append(obs)
        return True

    def negative_points(self, channel: ChannelId) -> tuple[Point, ...]:
        return tuple(obs.position for obs in self._negative.get(channel, ()))

    def snapshot(self) -> tuple[NegativeObservation, ...]:
        return tuple(obs for ch in sorted(self._negative) for obs in self._negative[ch])
```

- [ ] **Step 4: Run evidence tests**

Run: `pytest tests/test_cvr_evidence.py -q`

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```powershell
git add -- brl/cvr/evidence.py brl/cvr/__init__.py tests/test_cvr_evidence.py
git commit -m "feat: add CVR evidence ledger and future plan"
```

### Task 4: Implement independent Q3 and Q4 coverage certificates

**Files:**
- Create: `brl/cvr/certify.py`
- Create: `tests/test_cvr_certify.py`
- Modify: `brl/cvr/__init__.py`

- [ ] **Step 1: Write failing baseline, hole, and immutability tests**

```python
import numpy as np

from brl.coverage import s3_points, s25_points
from brl.cvr.certify import CoverageCertificateEngine


def test_original_plans_have_independent_geometric_certificates():
    q3 = CoverageCertificateEngine(3).certify(tuple(map(tuple, s3_points())))
    q4 = CoverageCertificateEngine(4).certify(tuple(map(tuple, s25_points())))
    assert q3.covered and q3.max_gap_m <= 0.0
    assert q4.covered and q4.max_gap_m <= 0.0


def test_q3_rejects_two_or_three_outer_points():
    points = ((0.0, 0.0), (1200.0, 0.0), (-600.0, 1039.23), (-600.0, -1039.23))
    result = CoverageCertificateEngine(3).certify(points)
    assert not result.covered
    assert result.max_gap_m > 0.0


def test_q4_rejects_plan_with_a_missing_outer_witness():
    points = tuple(map(tuple, np.delete(s25_points(), 13, axis=0)))
    result = CoverageCertificateEngine(4).certify(points)
    assert not result.covered
```

- [ ] **Step 2: Run tests and verify the module is missing**

Run: `pytest tests/test_cvr_certify.py -v`

Expected: FAIL on import.

- [ ] **Step 3: Implement Q3 maximum uncovered radius and Q4 triangulation proof**

Use SciPy only for deterministic Voronoi/Delaunay topology. Evaluate Q3's maximum distance to the nearest measurement point at all interior Voronoi vertices and all target-circle lower-envelope breakpoints; include each point's antipodal boundary angle and all pairwise distance-equality intersections on the target circle.

```python
# brl/cvr/certify.py
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, QhullError, Voronoi

DOMAIN_RADIUS = 1800.0
MIN_RECEIVE_RADIUS = 1000.0


@dataclass(frozen=True)
class CertificateResult:
    covered: bool
    mode: int
    max_gap_m: float
    witness_count: int
    reason: str


def _nearest_distance(point: np.ndarray, points: np.ndarray) -> float:
    return float(np.min(np.linalg.norm(points - point[None, :], axis=1)))


def _boundary_candidate_angles(points: np.ndarray) -> list[float]:
    angles = []
    for p in points:
        if np.linalg.norm(p) > 1e-12:
            angles.append(math.atan2(p[1], p[0]) + math.pi)
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            normal = points[j] - points[i]
            rhs = (float(points[j] @ points[j]) - float(points[i] @ points[i])) / 2.0
            a, b = float(normal[0]), float(normal[1])
            amp = math.hypot(a, b) * DOMAIN_RADIUS
            if amp <= 1e-12 or abs(rhs) > amp + 1e-9:
                continue
            base = math.atan2(b, a)
            offset = math.acos(max(-1.0, min(1.0, rhs / amp)))
            angles.extend((base - offset, base + offset))
    return angles


def q3_required_radius(points: np.ndarray) -> float:
    candidates = [np.zeros(2)]
    if len(points) >= 3:
        try:
            vor = Voronoi(points)
            candidates.extend(v for v in vor.vertices
                              if np.linalg.norm(v) <= DOMAIN_RADIUS + 1e-9)
        except QhullError:
            pass
    for angle in _boundary_candidate_angles(points):
        candidates.append(DOMAIN_RADIUS * np.array([math.cos(angle), math.sin(angle)]))
    return max(_nearest_distance(np.asarray(q), points) for q in candidates)


def _hull_inradius_at_origin(hull: ConvexHull) -> float:
    # scipy hull facets satisfy normal @ x + offset <= 0 inside the hull.
    normals = hull.equations[:, :2]
    offsets = hull.equations[:, 2]
    if np.any(offsets > 1e-9):
        return -math.inf
    return float(np.min(-offsets / np.linalg.norm(normals, axis=1)))


def _origin_to_segment(a: np.ndarray, b: np.ndarray) -> float:
    edge = b - a
    t = float(np.clip(-a @ edge / (edge @ edge), 0.0, 1.0))
    return float(np.linalg.norm(a + t * edge))


def _triangle_intersects_domain(vertices: np.ndarray) -> bool:
    cross = []
    for i in range(3):
        a, b = vertices[i], vertices[(i + 1) % 3]
        cross.append(float(np.cross(b - a, -a)))
    origin_inside = all(v >= -1e-9 for v in cross) or all(v <= 1e-9 for v in cross)
    if origin_inside:
        return True
    return min(_origin_to_segment(vertices[i], vertices[(i + 1) % 3])
               for i in range(3)) <= DOMAIN_RADIUS + 1e-9


def q4_max_relevant_edge(points: np.ndarray) -> tuple[float, bool]:
    hull = ConvexHull(points)
    if _hull_inradius_at_origin(hull) < DOMAIN_RADIUS:
        return math.inf, False
    tri = Delaunay(points)
    maximum = 0.0
    for simplex in tri.simplices:
        vertices = points[simplex]
        if not _triangle_intersects_domain(vertices):
            continue
        maximum = max(maximum, max(float(np.linalg.norm(vertices[i]-vertices[j]))
                                   for i in range(3) for j in range(i)))
    return maximum, True


class CoverageCertificateEngine:
    def __init__(self, mode: int, margin_m: float = 1e-6):
        if mode not in (3, 4):
            raise ValueError("mode must be 3 or 4")
        self.mode = mode
        self.margin_m = margin_m

    def certify(self, points) -> CertificateResult:
        array = np.asarray(points, dtype=float)
        if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
            raise ValueError("certificate points must be finite 2D coordinates")
        try:
            required = (q3_required_radius(array) if self.mode == 3
                        else q4_max_relevant_edge(array)[0])
            covered = required <= MIN_RECEIVE_RADIUS - self.margin_m
            return CertificateResult(covered, self.mode,
                                     float(required - MIN_RECEIVE_RADIUS),
                                     len(array), "covered" if covered else "geometric_gap")
        except QhullError:
            return CertificateResult(False, self.mode, math.inf, len(array), "degenerate_hull")
```

Before accepting this implementation, add a dense adversarial check to the test file. For Q3, sample 720,000 deterministic polar points and assert their nearest distance does not exceed the analytic result plus `1e-6`. For Q4, sample every relevant Delaunay triangle on a barycentric grid and all 720 directions, asserting at least one triangle vertex is within 1000 meters and in the visible half-plane.

- [ ] **Step 4: Run focused certificate tests**

Run:

```powershell
pytest tests/test_cvr_certify.py -q
```

Expected: all certificate tests pass, including S3/S25 baseline certificates and both deliberate holes.

- [ ] **Step 5: Commit**

```powershell
git add -- brl/cvr/certify.py brl/cvr/__init__.py tests/test_cvr_certify.py
git commit -m "feat: add independent CVR coverage certificates"
```

### Task 5: Generate deterministic variable points and certified local replacements

**Files:**
- Create: `brl/cvr/waypoints.py`
- Create: `tests/test_cvr_waypoints.py`
- Modify: `brl/cvr/__init__.py`

- [ ] **Step 1: Write failing candidate purity and replacement tests**

```python
import numpy as np

from brl.coverage import s3_points
from brl.cvr.evidence import CoverageNode, FutureCoveragePlan
from brl.cvr.ids import ChannelId, PlanNodeId, StationId
from brl.cvr.waypoints import PublicCVRSnapshot, WaypointGenerator


def _q3_plan():
    channels = frozenset(ChannelId(c) for c in range(1, 21))
    return FutureCoveragePlan.initial(tuple(
        CoverageNode(PlanNodeId(f"fixed-{i}"), tuple(map(float, p)), channels, StationId(i))
        for i, p in enumerate(s3_points())))


def test_same_public_snapshot_produces_identical_candidates():
    snapshot = PublicCVRSnapshot(3, (0.0, 0.0), 1, (), _q3_plan())
    first = WaypointGenerator().generate(snapshot)
    second = WaypointGenerator().generate(snapshot)
    assert first == second


def test_replacement_never_reuses_a_legacy_station_id():
    snapshot = PublicCVRSnapshot(3, (0.0, 0.0), 1,
                                 ((ChannelId(1), (800.0, 400.0), 70.0),), _q3_plan())
    proposals = WaypointGenerator().replacement_proposals(snapshot)
    assert proposals
    assert all(node.legacy_station is None
               for proposal in proposals for node in proposal.added)
```

- [ ] **Step 2: Run tests and verify missing implementation**

Run: `pytest tests/test_cvr_waypoints.py -v`

Expected: FAIL on import.

- [ ] **Step 3: Implement public snapshots, finite candidates, and replacements**

```python
# brl/cvr/waypoints.py
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

from .evidence import CoverageNode, FutureCoveragePlan
from .ids import ChannelId, PlanNodeId


@dataclass(frozen=True)
class PublicCVRTrack:
    channel: ChannelId
    center: tuple[float, float]
    radius: float
    next_kind: str
    next_position: tuple[float, float]


@dataclass(frozen=True)
class PublicCVRSnapshot:
    mode: int
    position: tuple[float, float]
    current_channel: int
    tracks: tuple[PublicCVRTrack, ...]
    plan: FutureCoveragePlan


@dataclass(frozen=True)
class ReplacementProposal:
    proposal_id: str
    removed: tuple[PlanNodeId, ...]
    added: tuple[CoverageNode, ...]


class WaypointGenerator:
    def generate(self, snapshot: PublicCVRSnapshot) -> tuple[tuple[float, float], ...]:
        points = []
        current = np.asarray(snapshot.position)
        for track in snapshot.tracks:
            c = np.asarray(track.center)
            delta = c - current
            length = float(np.linalg.norm(delta))
            if length > 1e-9:
                u = delta / length
                v = np.array([-u[1], u[0]])
                for forward in (0.5, 0.75, 1.0):
                    for lateral in (-300.0, 0.0, 300.0):
                        q = current + forward * delta + lateral * v
                        points.append((float(q[0]), float(q[1])))
            points.append((float(c[0]), float(c[1])))
            points.append(track.next_position)
        for node in snapshot.plan.nodes[:2]:
            points.append(node.position)
        return tuple(sorted(set(points), key=lambda p: (round(p[0], 6), round(p[1], 6))))

    def replacement_proposals(self, snapshot: PublicCVRSnapshot) -> tuple[ReplacementProposal, ...]:
        candidates = self.generate(snapshot)
        nodes = snapshot.plan.nodes
        proposals = []
        serial = 0
        for width in (1, 2):
            for start in range(max(0, len(nodes) - width + 1)):
                removed = tuple(n.node_id for n in nodes[start:start+width])
                channels = frozenset(c for n in nodes[start:start+width] for c in n.channels)
                for point in candidates:
                    serial += 1
                    added = CoverageNode(PlanNodeId(f"var-v{snapshot.plan.version}-{serial}"),
                                         point, channels, None)
                    proposals.append(ReplacementProposal(f"r{serial}", removed, (added,)))
        return tuple(proposals[:96])
```

The first implementation deliberately proposes only one variable point per removed block. A later code task may add a two-point replacement only if one-point replacements cannot pass the certificate; do not add continuous optimization in this task.

- [ ] **Step 4: Run waypoint tests**

Run: `pytest tests/test_cvr_waypoints.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add -- brl/cvr/waypoints.py brl/cvr/__init__.py tests/test_cvr_waypoints.py
git commit -m "feat: generate deterministic CVR waypoint replacements"
```

### Task 6: Add certified plan mutation and bounded segment planning

**Files:**
- Create: `brl/cvr/planner.py`
- Create: `tests/test_cvr_planner.py`
- Modify: `brl/cvr/__init__.py`

- [ ] **Step 1: Write failing tests for certificate filtering, exit cost, and budgets**

```python
import pytest

from brl.coverage import s3_points
from brl.cvr.certify import CoverageCertificateEngine
from brl.cvr.evidence import CoverageNode, CoverageEvidenceLedger, FutureCoveragePlan
from brl.cvr.ids import ChannelId, PlanNodeId, StationId
from brl.cvr.planner import MacroTask, SegmentPlanner, SegmentPlannerConfig
from brl.cvr.waypoints import PublicCVRSnapshot, WaypointGenerator


@pytest.fixture
def q3_snapshot():
    channels = frozenset(ChannelId(c) for c in range(1, 21))
    plan = FutureCoveragePlan.initial(tuple(
        CoverageNode(PlanNodeId(f"fixed-{i}"), tuple(map(float, p)),
                     channels, StationId(i))
        for i, p in enumerate(s3_points())))
    return PublicCVRSnapshot(3, (0.0, 0.0), 1, (), plan)


def test_planner_rejects_cheaper_replacement_that_opens_a_coverage_hole(q3_snapshot):
    planner = SegmentPlanner(CoverageCertificateEngine(3), WaypointGenerator())
    chosen = planner.plan(q3_snapshot, CoverageEvidenceLedger(),
                          exit_position=(1200.0, 0.0))
    assert chosen.certificate.covered
    assert all(m.plan_after.version >= q3_snapshot.plan.version
               for m in chosen.mutations)


def test_segment_cost_includes_return_to_declared_exit(q3_snapshot):
    planner = SegmentPlanner(CoverageCertificateEngine(3), WaypointGenerator())
    task = MacroTask("measure", "n", (100.0, 0.0), (1, 2))
    near_exit = planner.segment_cost(q3_snapshot, (task,), (200.0, 0.0))
    far_exit = planner.segment_cost(q3_snapshot, (task,), (-1800.0, 0.0))
    assert far_exit > near_exit


def test_zero_planning_budget_returns_frozen_baseline_segment(q3_snapshot):
    config = SegmentPlannerConfig(max_call_s=0.0, max_total_s=0.0)
    planner = SegmentPlanner(CoverageCertificateEngine(3), WaypointGenerator(), config)
    segment = planner.plan(q3_snapshot, CoverageEvidenceLedger(),
                           exit_position=(1200.0, 0.0))
    assert segment.reason == "planning_budget_fallback"
    assert not segment.mutations
```

- [ ] **Step 2: Run tests and verify missing planner**

Run: `pytest tests/test_cvr_planner.py -v`

Expected: FAIL on import.

- [ ] **Step 3: Implement typed macro tasks and exact known-segment cost**

```python
# brl/cvr/planner.py
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np

from .certify import CertificateResult, CoverageCertificateEngine
from .evidence import CoverageEvidenceLedger, FutureCoveragePlan
from .waypoints import PublicCVRSnapshot, ReplacementProposal, WaypointGenerator


@dataclass(frozen=True)
class SegmentPlannerConfig:
    beam_width: int = 16
    max_depth: int = 4
    max_proposals: int = 96
    min_predicted_gain_s: float = 1.0
    max_call_s: float = 0.5
    max_total_s: float = 12.0


@dataclass(frozen=True)
class MacroTask:
    kind: str
    key: str
    position: tuple[float, float]
    channels: tuple[int, ...] = ()


@dataclass(frozen=True)
class PlanMutation:
    proposal_id: str
    plan_before: FutureCoveragePlan
    plan_after: FutureCoveragePlan


@dataclass(frozen=True)
class PlannedSegment:
    tasks: tuple[MacroTask, ...]
    mutations: tuple[PlanMutation, ...]
    certificate: CertificateResult
    channel_certificates: tuple[tuple[int, CertificateResult], ...]
    estimated_cost_s: float
    baseline_cost_s: float
    reason: str


class SegmentPlanner:
    def __init__(self, engine: CoverageCertificateEngine,
                 generator: WaypointGenerator,
                 config: SegmentPlannerConfig | None = None):
        self.engine = engine
        self.generator = generator
        self.config = config or SegmentPlannerConfig()
        self.total_wall_s = 0.0

    @staticmethod
    def _switches(current: int, channels: tuple[int, ...]) -> int:
        total, last = 0, current
        for channel in channels:
            total += int(channel != last)
            last = channel
        return total

    def segment_cost(self, snapshot: PublicCVRSnapshot,
                     tasks: tuple[MacroTask, ...], exit_position) -> float:
        position = np.asarray(snapshot.position, dtype=float)
        channel = snapshot.current_channel
        total = 0.0
        for task in tasks:
            q = np.asarray(task.position, dtype=float)
            total += float(np.linalg.norm(q-position)) / 5.0
            if task.kind == "measure":
                total += 5.0 * len(task.channels) + self._switches(channel, task.channels)
                if task.channels:
                    channel = task.channels[-1]
            position = q
        total += float(np.linalg.norm(np.asarray(exit_position)-position)) / 5.0
        return total

    def _certificates(self, snapshot: PublicCVRSnapshot,
                      plan: FutureCoveragePlan,
                      ledger: CoverageEvidenceLedger
                      ) -> tuple[tuple[int, CertificateResult], ...]:
        # Worst-case no-signal planning: a future node is a witness only for a
        # channel it actually promises to measure. Per-channel calls are made by
        # plan(); never merge channel evidence here.
        channels = sorted({c for n in plan.nodes for c in n.channels})
        results = []
        for channel in channels:
            points = (ledger.negative_points(channel)
                      + tuple(n.position for n in plan.nodes if channel in n.channels))
            results.append((channel.value, self.engine.certify(points)))
        return tuple(results)

    @staticmethod
    def _worst_certificate(mode: int,
                           results: tuple[tuple[int, CertificateResult], ...]
                           ) -> CertificateResult:
        if not results:
            return CertificateResult(True, mode, -1000.0, 0, "no_unknown_channels")
        return min((result for _, result in results),
                   key=lambda result: (result.covered, -result.max_gap_m))
```

Then complete `plan()` with these fixed rules:

1. Construct the unchanged baseline segment from the nearest pending plan node and declared exit.
2. Generate at most 96 replacement proposals.
3. Apply a proposal to an immutable plan.
4. For each still-unknown channel, certify `ledger.negative_points(channel) + future points assigned to channel`.
5. Reject any proposal with one failed channel certificate.
6. Convert the replacement into measure tasks; preserve deterministic current-channel-first ordering.
7. Expand at most four tasks, retain 16 lowest `(estimated_cost, stable_task_key)` partial segments per layer.
8. Accept a replacement only if it saves at least one second relative to the same declared exit.
9. If elapsed time exceeds 0.5 seconds or cumulative time exceeds 12 seconds, return the unchanged baseline segment with reason `planning_budget_fallback`.

Do not silently use the single `CertificateResult` combination shown above as the final all-channel test; `PlannedSegment` must also log a tuple of per-channel results so the auditor can identify any failed channel.

- [ ] **Step 4: Run planner tests**

Run: `pytest tests/test_cvr_planner.py -q`

Expected: all tests pass; a cheaper but uncertified plan is rejected, exit cost changes ranking, and zero budget returns the frozen plan.

- [ ] **Step 5: Commit**

```powershell
git add -- brl/cvr/planner.py brl/cvr/__init__.py tests/test_cvr_planner.py
git commit -m "feat: add certified bounded CVR segment planner"
```

### Task 7: Execute CVR segments with custom completion and exact ISR fallback

**Files:**
- Create: `brl/cvr/policy.py`
- Create: `tests/test_cvr_policy.py`
- Modify: `brl/cvr/__init__.py`

- [ ] **Step 1: Write failing end-to-end correctness tests**

```python
import numpy as np
import pytest

from brl.cvr import CVRCandidate, CVRConfig
from brl.local_env import RadioEnv, random_sources
from brl.protocol import ActionIOError


@pytest.mark.parametrize("mode,count", [(3, 10), (3, 16), (4, 10), (4, 16)])
def test_cvr_clears_all_sources_and_balances_cost(mode, count):
    seed = 7000 + mode * 100 + count
    sources = random_sources(mode, count, np.random.default_rng(seed))
    env = RadioEnv(mode=mode, n_sources=count, seed=seed, bearing_decimals=2)
    env.reset(seed=seed, sources=sources)
    result = CVRCandidate(mode).run(env)
    assert result["success"]
    assert env.cleared_count() == count
    reconstructed = (env.move_distance / 5 + 5 * env.n_measure + env.n_switch
                     + 3 * env.n_clear_fail + 5 * env.cleared_count())
    assert abs(env.virtual_time - reconstructed) <= 1e-6


def test_variable_measurement_never_claims_a_legacy_coverage_index(monkeypatch):
    env = RadioEnv(mode=3, n_sources=10, seed=7100, bearing_decimals=2)
    seen = []
    original = env.measure
    def capture(position, channel, coverage_idx=None, is_refine=False):
        seen.append((tuple(position), coverage_idx))
        return original(position, channel, coverage_idx, is_refine)
    monkeypatch.setattr(env, "measure", capture)
    CVRCandidate(3).run(env)
    assert all(idx is None or np.linalg.norm(np.asarray(p)-env.coverage_points[idx]) <= 1e-6
               for p, idx in seen)


def test_transport_error_stops_before_a_different_action(monkeypatch):
    env = RadioEnv(mode=4, n_sources=10, seed=7200, bearing_decimals=2)
    calls = []
    def fail(*args, **kwargs):
        calls.append((args, kwargs))
        raise ActionIOError("response uncertain")
    monkeypatch.setattr(env, "measure", fail)
    with pytest.raises(ActionIOError):
        CVRCandidate(4).run(env)
    assert len(calls) == 1


def test_zero_budget_falls_back_to_isr_and_clears():
    env = RadioEnv(mode=4, n_sources=10, seed=7300, bearing_decimals=2)
    cfg = CVRConfig(max_plan_call_s=0.0, max_plan_total_s=0.0)
    result = CVRCandidate(4, cfg).run(env)
    assert result["success"] and env.cleared_count() == 10
    assert result["fallback_reason"] in {"per_call_budget", "episode_budget"}
```

- [ ] **Step 2: Run tests and verify missing candidate**

Run: `pytest tests/test_cvr_policy.py -v`

Expected: FAIL because `CVRCandidate` is not exported.

- [ ] **Step 3: Implement CVR as a JSP execution reuse with a separate plan**

Use `JSPPolicy` only for `_observe_scan`, `_locator_action`, `_execute_locator_one`, `_resolve`, and `resume_baseline`; CVR owns search tasks and completion.

```python
# brl/cvr/policy.py
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np

from brl.coverage import s3_points, s25_points
from brl.independent_candidate import CheckedView
from brl.jsp.policy import JSPPolicy
from brl.protocol import CertificateViolation

from .certify import CoverageCertificateEngine
from .evidence import CoverageEvidenceLedger, CoverageNode, FutureCoveragePlan
from .ids import ChannelId, PlanNodeId, StationId
from .planner import SegmentPlanner, SegmentPlannerConfig
from .waypoints import PublicCVRSnapshot, WaypointGenerator


@dataclass(frozen=True)
class CVRConfig:
    planner_enabled: bool = True
    variable_waypoints: bool = True
    max_plan_call_s: float = 0.5
    max_plan_total_s: float = 12.0
    max_segments: int = 100


class CVRPolicy:
    def __init__(self, env, config: CVRConfig | None = None):
        self.env = env
        self.config = config or CVRConfig()
        self.executor = JSPPolicy(env)
        points = s3_points() if env.mode == 3 else s25_points()
        all_channels = frozenset(ChannelId(c) for c in range(1, 21))
        self.plan = FutureCoveragePlan.initial(tuple(
            CoverageNode(PlanNodeId(f"fixed-{i}"), tuple(map(float, p)),
                         all_channels, StationId(i))
            for i, p in enumerate(points)))
        self.ledger = CoverageEvidenceLedger()
        self.certified_absent: set[ChannelId] = set()
        pcfg = SegmentPlannerConfig(max_call_s=self.config.max_plan_call_s,
                                    max_total_s=self.config.max_plan_total_s)
        self.planner = SegmentPlanner(CoverageCertificateEngine(env.mode),
                                      WaypointGenerator(), pcfg)
        self.events = []
        self.fallback_reason = ""
        self._evidence_serial = 0

    def _complete(self) -> bool:
        if self.env.cleared_count() >= 16:
            return True
        return all(self.env.channels[c].status == "cleared"
                   or ChannelId(c) in self.certified_absent for c in range(1, 21))

    def _measure_node(self, node: CoverageNode, channel: ChannelId) -> None:
        self._evidence_serial += 1
        evidence_id = f"cvr-observation-{self._evidence_serial}"
        coverage_idx = None if node.legacy_station is None else node.legacy_station.value
        response = self.env.measure(node.position, channel.value,
                                    coverage_idx=coverage_idx)
        point = np.asarray(node.position, dtype=float)
        self.executor._observe_scan(channel.value, point, response)
        self.ledger.accept(channel, node.position,
                           response["measure_result"], evidence_id,
                           accepted=response.get("accepted") is True)
        self.plan = FutureCoveragePlan(
            self.plan.version + 1,
            tuple(updated for n in self.plan.nodes
                  for updated in (n.without(channel) if n.node_id == node.node_id else n,)
                  if updated.channels))

    def _fallback(self, reason: str) -> None:
        self.fallback_reason = reason
        self.executor.resume_baseline()
```

Complete `run()` with these rules:

1. Execute the origin node first so behavior remains comparable to ISR-v1.
2. Remove a discovered channel from every future coverage node because it no longer needs an absence proof; keep its locator.
3. Before applying a plan mutation, save the per-channel certificate tuple and plan version in an event.
4. Execute a planned segment task-by-task. For a fixed node pass its real legacy index; for a variable node pass `None`.
5. After each accepted measurement, append the result and immediate plan version to the event log.
6. Recompute actual absence using only accepted negative points; add a channel to `certified_absent` only when its actual evidence certifies the full target domain.
7. On `near` or a deterministic clear action, allow the executor to finish the clear before planning another segment.
8. On invalidated geometry, abandon only the unexecuted mutation and restore the last certified future plan.
9. On planning timeout, segment budget, or 100 segments, call `_fallback` without resetting executor tracks, locators, current channel, accepted fixed-station indices, or elapsed virtual time.
10. Return success only when `_complete()` is true; otherwise raise `CertificateViolation`.

The returned result must include `decision_log`, `plan_versions`, `replacement_candidates`, `certified_replacements`, `executed_replacements`, `restored_legacy_stations`, `physical_stations_removed`, `extra_measurements`, `replans`, `planning_wall_s`, `fallback_reason`, `virtual_time_s`, distance/action counters, and `optical_fallbacks`.

- [ ] **Step 4: Run CVR policy and existing protocol tests**

Run:

```powershell
pytest tests/test_cvr_policy.py tests/test_bilateral_state.py tests/test_jsp_protocol.py tests/test_official_transport.py -q
```

Expected: all tests pass. The four CVR smoke cases clear exactly their generated source count with residual at most `1e-6`.

- [ ] **Step 5: Commit**

```powershell
git add -- brl/cvr/policy.py brl/cvr/__init__.py tests/test_cvr_policy.py
git commit -m "feat: execute certified CVR route segments"
```

### Task 8: Add an independent certificate replay auditor

**Files:**
- Create: `scripts/audit_cvr_certificates.py`
- Create: `tests/test_cvr_audit.py`

- [ ] **Step 1: Write a failing tamper-detection test**

```python
import json
from pathlib import Path

from scripts.audit_cvr_certificates import audit_events


def test_auditor_rejects_future_point_as_completed_evidence(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps({
        "event": "certified_absent", "channel": 1,
        "accepted_negative_points": [],
        "future_points": [[0.0, 0.0]], "mode": 3,
        "plan_version": 2
    }) + "\n", encoding="utf-8")
    result = audit_events(path)
    assert not result["valid"]
    assert "future evidence" in result["errors"][0]
```

- [ ] **Step 2: Run the test and verify the import failure**

Run: `pytest tests/test_cvr_audit.py -v`

Expected: FAIL because the auditor does not exist.

- [ ] **Step 3: Implement replay from serialized evidence only**

```python
# scripts/audit_cvr_certificates.py
from __future__ import annotations
import argparse
import json
from pathlib import Path

from brl.cvr.certify import CoverageCertificateEngine


def audit_events(path: Path) -> dict:
    errors = []
    last_version = 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        version = int(row.get("plan_version", 0))
        if version < last_version:
            errors.append(f"line {line_no}: plan version moved backward")
        last_version = max(last_version, version)
        if row.get("event") == "certified_absent":
            if row.get("future_points"):
                errors.append(f"line {line_no}: future evidence used for completed absence")
                continue
            engine = CoverageCertificateEngine(int(row["mode"]))
            result = engine.certify(tuple(map(tuple, row["accepted_negative_points"])))
            if not result.covered:
                errors.append(f"line {line_no}: absence certificate does not replay")
    return {"valid": not errors, "errors": errors, "last_plan_version": last_version}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("events", type=Path)
    args = parser.parse_args()
    result = audit_events(args.events)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
```

Extend the auditor in the same task to check unique request IDs, node IDs, removed-node existence, increasing versions, legacy coordinate/index agreement, certificate-before-mutation ordering, and that every `certified_absent` event uses only accepted `no_signal` observations for the same channel.

- [ ] **Step 4: Run auditor and CVR tests**

Run: `pytest tests/test_cvr_audit.py tests/test_cvr_evidence.py tests/test_cvr_certify.py -q`

Expected: all tests pass and the deliberately tampered log is rejected.

- [ ] **Step 5: Commit**

```powershell
git add -- scripts/audit_cvr_certificates.py tests/test_cvr_audit.py
git commit -m "test: add independent CVR certificate auditor"
```

### Task 9: Add resumable paired development experiments and gates

**Files:**
- Create: `scripts/run_cvr_experiments.py`
- Create: `scripts/analyze_cvr_local.py`
- Create: `tests/test_cvr_experiment.py`

- [ ] **Step 1: Write failing schedule, persistence, and gate tests**

```python
from scripts.run_cvr_experiments import jobs, summarize


def test_development_schedule_has_exact_three_arms():
    rows = list(jobs("dev", [3, 4], ["ISR", "SEGFIXED", "CVR"]))
    assert len(rows) == 2 * 50 * 3
    assert {row[-1] for row in rows} == {"ISR", "SEGFIXED", "CVR"}


def test_candidate_below_three_percent_is_not_eligible():
    rows = paired_fixture(baseline=100.0, candidate=98.0, pairs=50)
    result = summarize(rows, "dev")
    assert result["development_selection"][0]["eligible"] == []


def test_mechanism_gate_requires_removed_station_or_lower_total_cost():
    rows = paired_fixture(baseline=100.0, candidate=95.0, pairs=50,
                          physical_stations_removed=0, mean_distance_delta_m=20.0)
    result = summarize(rows, "dev")
    assert not result["development_selection"][0]["mechanism_gate"]
```

Define `paired_fixture` in the test file with complete success, source count, cost residual, tail, runtime, and mechanism fields so each failing gate can be isolated.

- [ ] **Step 2: Run tests and verify missing runner**

Run: `pytest tests/test_cvr_experiment.py -v`

Expected: FAIL on import.

- [ ] **Step 3: Implement the incremental runner**

Copy the proven persistence structure from `scripts/run_jsp_experiments.py`, then make these explicit changes:

```python
def jobs(stage: str, modes: list[int], arms: list[str]):
    groups = ({"dev": ((160000, 50, "spatial"),),
               "acceptance": ((280000, 200, "spatial"),
                              (281000, 30, "endpoints"),
                              (281100, 30, "constant"))})[stage]
    for mode in modes:
        for start, count, error_mode in groups:
            for offset in range(count):
                seed = start + offset
                kind = "edge" if offset % 3 == 0 else "uniform"
                for arm in arms:
                    yield mode, seed, kind, error_mode, arm


def make_candidate(mode: int, arm: str):
    if arm == "ISR":
        return IndependentCandidate(mode)
    if arm == "SEGFIXED":
        return CVRCandidate(mode, CVRConfig(variable_waypoints=False))
    if arm == "CVR":
        return CVRCandidate(mode)
    raise ValueError(arm)
```

Add `variable_waypoints: bool = True` to `CVRConfig`; when false, the segment planner may change task grouping/order but must reject every `ReplacementProposal`.

Persist each run before beginning with status `started`, then atomically append its final `complete` or `failed` record. Use a stable run key `(mode, seed, error_mode, arm, code_hash, config_hash)`. Never overwrite a completed record. Store decisions in a separate per-run JSONL path so an interrupted write cannot corrupt every run.

Fields must include all JSP cost fields plus:

```python
MECHANISM_FIELDS = [
    "replacement_candidates", "certified_replacements", "executed_replacements",
    "restored_legacy_stations", "physical_stations_removed", "extra_measurements",
    "replans", "segment_interruptions", "certificate_wall_s", "fallback_reason"
]
```

Development eligibility requires all clear, residual at most `1e-6`, mean improvement at least 3%, P95/max at most 103% of baseline, runtime P95 within `max(2*baseline, 10s)`, and either positive mean physical stations removed or negative mean distance and total cost deltas. Report SEG-FIXED and CVR independently; do not select CVR merely because it beats SEG-FIXED if it fails ISR-v1 gates.

The runner must refuse `--stage acceptance` unless passed a frozen development `summary.json` whose selected mode/candidate has every development gate true. It must also verify the acceptance seed ranges do not appear in any existing manifest under `results/` before creating its own manifest.

- [ ] **Step 4: Implement the analyzer with explicit cost decomposition**

`scripts/analyze_cvr_local.py` reads only complete, uniquely keyed run records and writes `summary.json`, `cost_decomposition.json`, and `mechanism.json`. Use:

```python
costs = {
    "move_s_per_source": distance_m / 5.0 / sources,
    "measure_s_per_source": 5.0 * measure_calls / sources,
    "switch_s_per_source": switches / sources,
    "failed_clear_s_per_source": 3.0 * failed_clear / sources,
    "success_clear_s_per_source": 5.0 * cleared / sources,
}
```

Pair by mode, seed, kind, and error mode. Bootstrap 10,000 paired mean deltas with seed `20260912`; report 95% and 97.5% intervals, P90/P95/max, aggregate `sum(T)/sum(N)`, actual-wall P95, and every mechanism field.

- [ ] **Step 5: Run experiment unit tests**

Run: `pytest tests/test_cvr_experiment.py -q`

Expected: schedule, persistence, duplicate, failure, seed-protection, mechanism, and statistical gate tests all pass.

- [ ] **Step 6: Commit**

```powershell
git add -- scripts/run_cvr_experiments.py scripts/analyze_cvr_local.py tests/test_cvr_experiment.py brl/cvr/policy.py
git commit -m "feat: add paired CVR local experiment runner"
```

### Task 10: Document, verify, and run the fixed development experiment

**Files:**
- Create: `docs/CVR_RUNBOOK.md`
- Generated: `results/cvr/jsp_bugfix_diagnostic_20260913/`
- Generated: `results/cvr/dev_160000_160049/`
- Generated: `results/cvr/development_report.md`

- [ ] **Step 1: Write the runbook before starting expensive runs**

Document the exact environment capture and commands:

```powershell
python --version
python -c "import numpy, scipy; print(numpy.__version__, scipy.__version__)"
git rev-parse HEAD
pytest -q
python scripts/run_cvr_experiments.py --stage dev --modes 3,4 --arms ISR,SEGFIXED,CVR --out-dir results/cvr/dev_160000_160049
python scripts/audit_cvr_certificates.py results/cvr/dev_160000_160049/decisions.jsonl
python scripts/analyze_cvr_local.py --runs results/cvr/dev_160000_160049/runs.csv --out-dir results/cvr/dev_160000_160049
```

Explain that no official experiment is authorized by the runbook and that acceptance is conditional on the generated development gate.

- [ ] **Step 2: Run all tests before experiments**

Run:

```powershell
pytest -q
```

Expected: all repository tests pass; record the exact count and elapsed time in the development manifest.

- [ ] **Step 3: Run the JSP bug-fix diagnostic**

Use only ISR, corrected JSP-G, and corrected JSP-L on the already-used development seeds. Preserve the original frozen summary and write a separate manifest naming commit hashes before and after the fix.

```powershell
python scripts/run_jsp_experiments.py --stage dev --modes 3,4 --arms ISR,JSPG,JSPL `
  --model-q3 results/jsp/models_20260913/jsp_q3.joblib `
  --model-q4 results/jsp/models_20260913/jsp_q4.joblib `
  --out-dir results/cvr/jsp_bugfix_diagnostic_20260913
```

Expected: 300 complete runs or an incrementally persisted failure. Report the effect, but label it diagnostic and keep ISR-v1 regardless of outcome.

- [ ] **Step 4: Run the fixed 300-run CVR development matrix**

```powershell
python scripts/run_cvr_experiments.py --stage dev --modes 3,4 `
  --arms ISR,SEGFIXED,CVR --out-dir results/cvr/dev_160000_160049
```

Expected: 50 paired scenarios per mode and arm, 300 complete runs, or immediate stop after a persisted correctness failure.

- [ ] **Step 5: Audit every certificate transition**

Run the auditor over every per-run decision JSONL, aggregate results, and require zero invalid logs. If one log fails, mark that CVR arm ineligible and do not repair then continue on the same development output directory; create a versioned diagnostic directory after the fix.

- [ ] **Step 6: Analyze the development gates**

Generate and inspect:

- per-mode ISR/SEG-FIXED/CVR mean `T/N`;
- paired 95% and 97.5% intervals;
- P90/P95/max and actual runtime;
- move/measure/switch/clear decomposition;
- physical stations removed, extra measurements, certified/executed replacements, replan and fallback counts;
- explicit eligibility outcome for Q3 and Q4.

Expected decision: each mode yields either one eligible frozen candidate or `null`. Do not continue to acceptance if the result is `null`.

- [ ] **Step 7: Write the evidence-based development report**

`results/cvr/development_report.md` must state:

1. whether the JSP bug fix changed the negative conclusion;
2. whether SEG-FIXED reduced short-horizon route bouncing;
3. whether CVR actually removed physical stations;
4. whether movement savings exceeded extra measurement/switching cost;
5. all-clear denominators and certificate-audit status;
6. Q3 and Q4 development eligibility;
7. that no independent acceptance or official test has been run unless a later artifact proves otherwise.

- [ ] **Step 8: Run verification once more after the report**

```powershell
pytest -q
git diff --check
git status --short
```

Expected: tests pass; no whitespace errors; only known generated evidence or unrelated pre-existing files remain untracked.

- [ ] **Step 9: Commit code, runbook, and compact machine-readable summaries**

Do not commit large per-action JSONL unless repository policy already tracks equivalent evidence. Commit source, tests, runbook, manifests, `summary.json`, `cost_decomposition.json`, `mechanism.json`, and the development report with explicit paths:

```powershell
git add -- brl/cvr tests/test_cvr_*.py scripts/run_cvr_experiments.py `
  scripts/audit_cvr_certificates.py scripts/analyze_cvr_local.py docs/CVR_RUNBOOK.md `
  results/cvr/dev_160000_160049/manifest.json `
  results/cvr/dev_160000_160049/summary.json `
  results/cvr/dev_160000_160049/cost_decomposition.json `
  results/cvr/dev_160000_160049/mechanism.json `
  results/cvr/development_report.md
git commit -m "feat: validate certified variable waypoint routing"
```

### Task 11: Conditional independent acceptance

**Files:**
- Generated only after an eligible development result: `results/cvr/acceptance_q3_20260913/` and/or `results/cvr/acceptance_q4_20260913/`
- Modify after completion: `results/cvr/development_report.md`

- [ ] **Step 1: Verify the development gate before creating output**

Read `development_selection` in `results/cvr/dev_160000_160049/summary.json`. Run exactly one of the following commands for each mode whose `preferred` field is non-null; substitute nothing inside a command.

```powershell
python scripts/run_cvr_experiments.py --stage acceptance --modes 3 --arms ISR,CVR `
  --development-summary results/cvr/dev_160000_160049/summary.json `
  --out-dir results/cvr/acceptance_q3_20260913
python scripts/run_cvr_experiments.py --stage acceptance --modes 3 --arms ISR,SEGFIXED `
  --development-summary results/cvr/dev_160000_160049/summary.json `
  --out-dir results/cvr/acceptance_q3_20260913
python scripts/run_cvr_experiments.py --stage acceptance --modes 4 --arms ISR,CVR `
  --development-summary results/cvr/dev_160000_160049/summary.json `
  --out-dir results/cvr/acceptance_q4_20260913
python scripts/run_cvr_experiments.py --stage acceptance --modes 4 --arms ISR,SEGFIXED `
  --development-summary results/cvr/dev_160000_160049/summary.json `
  --out-dir results/cvr/acceptance_q4_20260913
```

Choose the command whose mode and candidate exactly equal the `preferred` result. Expected before any run: the command validates that exact candidate, a matching code/config hash, and unused acceptance seeds. The runner rejects the other three commands when they do not match the summary. If no candidate passed development, do not run this task and record `independent_acceptance.started=false`.

- [ ] **Step 2: Execute one frozen candidate per eligible mode**

For each eligible mode, run exactly:

- spatial error `280000–280199`: 200 ISR/candidate pairs;
- endpoint ±1° `281000–281029`: 30 pairs;
- constant +1° `281100–281129`: 30 pairs.

Expected: 520 runs per eligible mode, with immediate persisted stop on incomplete clearance, certificate failure, or cost residual above `1e-6`.

- [ ] **Step 3: Apply the frozen acceptance gates**

Require all runs to clear all actual sources and pass certificate replay; spatial mean `T/N` at least 3% below ISR; paired 97.5% interval upper bound below zero; spatial P95/max at most 103% of ISR; each pressure mean no more than 3% worse; wall P95 at most `max(2*ISR,10s)`.

- [ ] **Step 4: Record the decision without adding samples**

If any gate fails, recommend ISR-v1 and do not add scenes. If a correctness bug appears, preserve the failed acceptance result; a repaired version cannot use the viewed seed ranges as independent success evidence.

- [ ] **Step 5: Commit compact acceptance evidence if it ran**

```powershell
git add -- results/cvr/acceptance_q3_20260913/manifest.json `
  results/cvr/acceptance_q3_20260913/summary.json `
  results/cvr/acceptance_q3_20260913/cost_decomposition.json `
  results/cvr/acceptance_q3_20260913/mechanism.json `
  results/cvr/acceptance_q4_20260913/manifest.json `
  results/cvr/acceptance_q4_20260913/summary.json `
  results/cvr/acceptance_q4_20260913/cost_decomposition.json `
  results/cvr/acceptance_q4_20260913/mechanism.json results/cvr/development_report.md
git commit -m "test: record CVR independent acceptance"
```

Stage only the acceptance directory or directories that actually exist; omission of an ineligible mode is expected and must be recorded in the report.

Do not start or prepare an official CVR batch in this plan. An official plan is a separate decision after independent acceptance and after verifying simulator availability and remaining opportunities.
