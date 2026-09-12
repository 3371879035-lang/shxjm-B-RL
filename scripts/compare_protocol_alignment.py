"""Create machine-readable exact comparison of two ISR protocol replays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    reference = pd.read_csv(args.reference)
    current = pd.read_csv(args.current)
    keys = ["mode", "seed", "kind"]
    if not reference[keys].equals(current[keys]):
        raise RuntimeError("protocol replay keys do not align")
    fields = {}
    for column in reference.columns:
        if column in keys:
            continue
        left = reference[column].to_numpy()
        right = current[column].to_numpy()
        if left.dtype.kind in "OUSb":
            equal = bool(np.array_equal(left, right, equal_nan=True))
            maximum = None
        else:
            delta = np.abs(left.astype(float) - right.astype(float))
            maximum = float(np.nanmax(delta)) if len(delta) else 0.0
            equal = bool(np.allclose(left, right, rtol=0.0, atol=1e-9, equal_nan=True))
        fields[column] = {"equal": equal, "max_abs_delta": maximum}
    result = {
        "rows": len(reference), "keys_equal": True,
        "all_fields_equal_with_atol_1e-9": all(item["equal"] for item in fields.values()),
        "fields": fields,
    }
    path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
