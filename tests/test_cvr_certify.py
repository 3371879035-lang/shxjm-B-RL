import math

import numpy as np

from brl.coverage import s25_points, s3_points
from brl.cvr.certify import CoverageCertificateEngine, q3_required_radius


def test_original_plans_have_independent_geometric_certificates():
    q3 = CoverageCertificateEngine(3).certify(tuple(map(tuple, s3_points())))
    q4 = CoverageCertificateEngine(4).certify(tuple(map(tuple, s25_points())))
    assert q3.covered and q3.max_gap_m <= 0.0
    assert q4.covered and q4.max_gap_m <= 0.0


def test_q3_rejects_sparse_outer_points():
    points = ((0.0, 0.0), (1200.0, 0.0), (-600.0, 1039.23), (-600.0, -1039.23))
    result = CoverageCertificateEngine(3).certify(points)
    assert not result.covered
    assert result.max_gap_m > 0.0


def test_q4_rejects_plan_with_a_missing_outer_witness():
    points = tuple(map(tuple, np.delete(s25_points(), 13, axis=0)))
    result = CoverageCertificateEngine(4).certify(points)
    assert not result.covered


def test_q3_analytic_radius_dominates_dense_deterministic_sample():
    points = np.asarray(s3_points(), dtype=float)
    required = q3_required_radius(points)
    # Dense polar sampling is an independent lower-bound check on the maximum.
    angles = np.linspace(0.0, 2.0 * math.pi, 7200, endpoint=False)
    radii = np.linspace(0.0, 1800.0, 101)
    maximum = 0.0
    for radius in radii:
        sample = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
        nearest = np.min(np.linalg.norm(sample[:, None, :] - points[None, :, :], axis=2), axis=1)
        maximum = max(maximum, float(np.max(nearest)))
    assert maximum <= required + 1e-6
