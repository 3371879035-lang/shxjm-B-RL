"""Validated scikit-learn cost ranker for JSP."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import joblib
import numpy as np

from .snapshot import FEATURE_NAMES, FEATURE_VERSION


class JSPModelError(RuntimeError):
    pass


class JSPRanker:
    def __init__(self, model_path: str | Path, mode: int):
        self.path = Path(model_path)
        meta_path = self.path.with_suffix(self.path.suffix + ".json")
        if not self.path.is_file() or not meta_path.is_file():
            raise JSPModelError("model or metadata file is missing")
        self.metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if self.metadata.get("sha256") != digest:
            raise JSPModelError("model hash mismatch")
        if self.metadata.get("feature_version") != FEATURE_VERSION:
            raise JSPModelError("model feature version mismatch")
        if int(self.metadata.get("mode", -1)) != int(mode):
            raise JSPModelError("model problem mode mismatch")
        if tuple(self.metadata.get("feature_names", FEATURE_NAMES)) != FEATURE_NAMES:
            raise JSPModelError("model feature names mismatch")
        try:
            self.model = joblib.load(self.path)
            self.feature_min = np.asarray(self.metadata["feature_min"], dtype=float)
            self.feature_max = np.asarray(self.metadata["feature_max"], dtype=float)
        except Exception as exc:
            raise JSPModelError("model payload or metadata is invalid") from exc
        if (self.feature_min.shape != (len(FEATURE_NAMES),)
                or self.feature_max.shape != self.feature_min.shape
                or not np.all(np.isfinite(self.feature_min))
                or not np.all(np.isfinite(self.feature_max))
                or np.any(self.feature_min > self.feature_max)):
            raise JSPModelError("model feature range is invalid")
        if int(getattr(self.model, "n_features_in_", len(FEATURE_NAMES))) != len(FEATURE_NAMES):
            raise JSPModelError("estimator feature count mismatch")

    def predict(self, rows: np.ndarray, timeout_s: float = 0.25) -> np.ndarray:
        x = np.asarray(rows, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.feature_min):
            raise JSPModelError("invalid model feature shape")
        slack = 1e-9 + 0.05 * np.maximum(1.0, self.feature_max - self.feature_min)
        if np.any(x < self.feature_min - slack) or np.any(x > self.feature_max + slack):
            raise JSPModelError("public feature outside calibrated range")
        started = time.perf_counter()
        try:
            pred = np.asarray(self.model.predict(x), dtype=float)
        except Exception as exc:
            raise JSPModelError("model inference failed") from exc
        if time.perf_counter() - started > timeout_s:
            raise JSPModelError("model inference exceeded budget")
        if pred.shape != (len(x),) or not np.all(np.isfinite(pred)):
            raise JSPModelError("model produced invalid prediction")
        return pred
