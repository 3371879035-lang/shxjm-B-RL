from __future__ import annotations

import hashlib
import json

import joblib
import numpy as np
import pytest

from brl.jsp.model import JSPModelError, JSPRanker
from brl.jsp.snapshot import FEATURE_NAMES, FEATURE_VERSION


class ConstantModel:
    def predict(self, rows):
        return np.full(len(rows), -2.5)


def make_model(tmp_path, *, mode=4, sha_override=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "ranker.joblib"
    joblib.dump(ConstantModel(), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {
        "sha256": sha_override or digest,
        "feature_version": FEATURE_VERSION,
        "mode": mode,
        "feature_names": FEATURE_NAMES,
        "feature_min": [-1.0] * len(FEATURE_NAMES),
        "feature_max": [1.0] * len(FEATURE_NAMES),
    }
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(metadata))
    return path


def test_ranker_validates_and_predicts(tmp_path):
    path = make_model(tmp_path)
    ranker = JSPRanker(path, 4)
    assert ranker.predict(np.zeros((1, len(FEATURE_NAMES)))).tolist() == [-2.5]


def test_ranker_rejects_hash_and_mode(tmp_path):
    with pytest.raises(JSPModelError, match="hash"):
        JSPRanker(make_model(tmp_path, sha_override="bad"), 4)
    path = make_model(tmp_path / "second", mode=3)
    with pytest.raises(JSPModelError, match="mode"):
        JSPRanker(path, 4)


def test_ranker_rejects_out_of_range_and_shape(tmp_path):
    ranker = JSPRanker(make_model(tmp_path), 4)
    with pytest.raises(JSPModelError, match="range"):
        ranker.predict(np.full((1, len(FEATURE_NAMES)), 3.0))
    with pytest.raises(JSPModelError, match="shape"):
        ranker.predict(np.zeros((1, 3)))
