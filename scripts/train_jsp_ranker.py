"""Fit the frozen per-problem JSP cost ranker."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from brl.jsp.snapshot import FEATURE_NAMES, FEATURE_VERSION


def calibration_ranking(rows, predictions):
    groups = {}
    for row, pred in zip(rows, predictions):
        key = (row["seed"], row["trajectory"], row["snapshot_index"])
        groups.setdefault(key, []).append((float(row["delta_to_baseline_s"]), float(pred)))
    selected_deltas = []
    regrets = []
    concordant = comparable = 0
    for values in groups.values():
        actual = np.asarray([v[0] for v in values])
        pred = np.asarray([v[1] for v in values])
        chosen = int(np.argmin(pred))
        selected_deltas.append(float(actual[chosen]))
        regrets.append(float(actual[chosen] - actual.min()))
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                if abs(actual[i] - actual[j]) <= 1e-9:
                    continue
                comparable += 1
                concordant += int((pred[i] - pred[j]) * (actual[i] - actual[j]) > 0)
    return {
        "snapshot_groups": len(groups),
        "pairwise_ranking_accuracy": (concordant / comparable if comparable else None),
        "mean_selected_delta_s": float(np.mean(selected_deltas)),
        "mean_selected_regret_s": float(np.mean(regrets)),
        "p95_selected_regret_s": float(np.percentile(regrets, 95)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, nargs="+")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--modes", default="3,4")
    args = parser.parse_args()
    modes = [int(value) for value in args.modes.split(",")]
    if not modes or any(mode not in (3, 4) for mode in modes):
        raise SystemExit("--modes must contain 3 or 4")
    rows = []
    data_hashes = {}
    for data_name in args.data:
        path = Path(data_name)
        rows.extend(json.loads(line) for line in path.open(encoding="utf-8"))
        data_hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    if any(not row.get("success") or abs(float(row["cost_identity_error_s"])) > 1e-6
           for row in rows):
        raise RuntimeError("training data contains an incomplete or invalid branch")
    if any(row.get("feature_version") != FEATURE_VERSION for row in rows):
        raise RuntimeError("training data feature version mismatch")
    keys = [(row["mode"], row["split"], row["seed"], row["trajectory"],
             row["snapshot_index"], row["candidate_index"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("training data contains duplicate branch keys")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = {"feature_version": FEATURE_VERSION, "feature_names": FEATURE_NAMES,
              "data_sha256": data_hashes,
              "training_script_sha256": script_hash,
              "python": sys.version, "numpy": np.__version__,
              "scikit_learn": sklearn.__version__, "modes": {}}
    for mode in modes:
        train = [r for r in rows if r["mode"] == mode and r["split"] == "train"]
        calibration = [r for r in rows if r["mode"] == mode and r["split"] == "calibration"]
        if not train or not calibration:
            raise RuntimeError(f"mode {mode} lacks frozen train/calibration rows")
        x = np.asarray([r["features"] for r in train], dtype=float)
        y = np.asarray([r["delta_to_baseline_s"] for r in train], dtype=float)
        case_counts = {}
        for r in train:
            key = (r["seed"], r["trajectory"])
            case_counts[key] = case_counts.get(key, 0) + 1
        weight = np.asarray([1.0 / case_counts[(r["seed"], r["trajectory"])] for r in train])
        model = HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.05, max_iter=160,
            max_leaf_nodes=15, min_samples_leaf=32, l2_regularization=1.0,
            random_state=20260913, early_stopping=False)
        model.fit(x, y, sample_weight=weight)
        cx = np.asarray([r["features"] for r in calibration], dtype=float)
        cy = np.asarray([r["delta_to_baseline_s"] for r in calibration], dtype=float)
        pred = np.asarray(model.predict(cx), dtype=float)
        model_path = out / f"jsp_q{mode}.joblib"
        joblib.dump(model, model_path)
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        all_x = np.vstack((x, cx))
        outside = np.any((cx < x.min(axis=0)) | (cx > x.max(axis=0)), axis=1)
        ranking = calibration_ranking(calibration, pred)
        metadata = {
            "feature_version": FEATURE_VERSION, "mode": mode, "sha256": digest,
            "feature_names": FEATURE_NAMES,
            "data_sha256": data_hashes, "training_script_sha256": script_hash,
            "feature_min": all_x.min(axis=0).tolist(), "feature_max": all_x.max(axis=0).tolist(),
            "train_rows": len(train), "calibration_rows": len(calibration),
            "calibration_mae_s": float(np.mean(np.abs(pred - cy))),
            "calibration_rmse_s": float(np.sqrt(np.mean((pred - cy) ** 2))),
            "calibration_bias_s": float(np.mean(pred - cy)),
            "calibration_rows_outside_train_fraction": float(np.mean(outside)),
            "calibration_ranking": ranking,
            "config": {"loss": "squared_error", "learning_rate": 0.05, "max_iter": 160,
                       "max_leaf_nodes": 15, "min_samples_leaf": 32,
                       "l2_regularization": 1.0, "random_state": 20260913,
                       "early_stopping": False},
        }
        model_path.with_suffix(model_path.suffix + ".json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        report["modes"][str(mode)] = metadata
    (out / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
