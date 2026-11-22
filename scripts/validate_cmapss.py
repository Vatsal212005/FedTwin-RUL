import argparse
import json
from pathlib import Path
import numpy as np

def check_split(path):
    data = np.load(path)
    required = {"X", "twin_aux", "y", "unit_ids", "regime_ids"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"{path} missing {sorted(missing)}")

    X = data["X"]
    aux = data["twin_aux"]
    y = data["y"]
    units = data["unit_ids"]
    regimes = data["regime_ids"]

    assert X.ndim == 3
    assert aux.ndim == 2 and aux.shape[1] == 5
    assert y.ndim == 1
    assert len(X) == len(aux) == len(y) == len(units) == len(regimes)
    assert np.isfinite(X).all()
    assert np.isfinite(aux).all()
    assert np.isfinite(y).all()

    return {
        "samples": len(X),
        "shape": tuple(X.shape),
        "units": len(np.unique(units)),
        "regimes": np.unique(regimes).tolist(),
        "y_min": float(y.min()),
        "y_max": float(y.max()),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="FD004")
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args()

    d = Path(args.processed_dir)
    train_path = d / f"{args.subset}_train.npz"
    test_path = d / f"{args.subset}_test.npz"
    meta_path = d / f"{args.subset}_meta.json"

    print("TRAIN:", check_split(train_path))
    print("TEST: ", check_split(test_path))

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print("Sensors:", meta["selected_sensor_ids"])
        print("Training-window stride:", meta.get("train_window_stride", "UNKNOWN"))
        print("Leakage guards:")
        for item in meta["leakage_guards"]:
            print(" -", item)

    print("Validation passed.")

if __name__ == "__main__":
    main()
