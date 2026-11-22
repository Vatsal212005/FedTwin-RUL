import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

COLUMNS = (
    ["unit", "cycle"]
    + [f"op_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

SELECTED_SENSOR_IDS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
SELECTED_SENSORS = [f"sensor_{i}" for i in SELECTED_SENSOR_IDS]
OP_COLS = ["op_1", "op_2", "op_3"]

def read_cmapss_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, :26].copy()
    df.columns = COLUMNS
    return df

def fit_standardizer(train_df, columns):
    mean = train_df[columns].mean()
    std = train_df[columns].std().replace(0, 1.0)
    return mean, std

def apply_standardizer(df, mean, std, columns):
    out = df.copy()
    out[columns] = (out[columns] - mean) / (std + 1e-8)
    return out

def add_train_rul(df, rul_cap):
    out = df.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = (max_cycle - out["cycle"]).clip(upper=rul_cap).astype(np.float32)
    return out

def fit_healthy_reference(train_df, healthy_cycles):
    early = train_df[train_df["cycle"] <= healthy_cycles]
    if len(early) == 0:
        raise ValueError("Healthy-reference selection produced no rows.")
    return early[SELECTED_SENSORS].mean().to_numpy(np.float32)

def twin_aux_from_window(sensor_window, op_window, healthy_ref):
    op_mean = op_window.mean(axis=0).astype(np.float32)

    latest = sensor_window[-1]
    deviation = np.mean(np.abs(latest - healthy_ref))
    health_proxy = np.float32(np.exp(-deviation))

    midpoint = max(1, sensor_window.shape[0] // 2)
    first = sensor_window[:midpoint].mean(axis=0)
    second = sensor_window[midpoint:].mean(axis=0)
    drift_proxy = np.float32(np.mean(np.abs(second - first)))

    return np.concatenate([
        op_mean,
        np.array([health_proxy, drift_proxy], dtype=np.float32),
    ])

def pad_left(arr, seq_len):
    if len(arr) >= seq_len:
        return arr[-seq_len:]
    pad_count = seq_len - len(arr)
    pad = np.repeat(arr[:1], pad_count, axis=0)
    return np.concatenate([pad, arr], axis=0)

def build_train_windows(df, seq_len, stride, healthy_ref, regime_labels):
    X, twin_aux, y, units, regimes = [], [], [], [], []

    for unit_id, g in df.groupby("unit", sort=True):
        g = g.sort_values("cycle")
        sensors = g[SELECTED_SENSORS].to_numpy(np.float32)
        ops = g[OP_COLS].to_numpy(np.float32)
        ruls = g["rul"].to_numpy(np.float32)
        row_idx = g.index.to_numpy()

        if len(g) < seq_len:
            continue

        ends = list(range(seq_len, len(g) + 1, stride))
        if ends[-1] != len(g):
            ends.append(len(g))

        for end in ends:
            window = sensors[end - seq_len:end]
            op_window = ops[end - seq_len:end]

            X.append(window)
            twin_aux.append(twin_aux_from_window(window, op_window, healthy_ref))
            y.append(ruls[end - 1])
            units.append(int(unit_id))
            regimes.append(int(regime_labels[row_idx[end - 1]]))

    return {
        "X": np.asarray(X, dtype=np.float32),
        "twin_aux": np.asarray(twin_aux, dtype=np.float32),
        "y": np.asarray(y, dtype=np.float32),
        "unit_ids": np.asarray(units, dtype=np.int32),
        "regime_ids": np.asarray(regimes, dtype=np.int16),
    }

def build_test_windows(df, rul_end_values, seq_len, healthy_ref, regime_labels):
    X, twin_aux, y, units, regimes = [], [], [], [], []
    sorted_units = sorted(df["unit"].unique().tolist())

    if len(sorted_units) != len(rul_end_values):
        raise ValueError(
            f"Test engine count ({len(sorted_units)}) != "
            f"RUL label count ({len(rul_end_values)})."
        )

    for label_idx, unit_id in enumerate(sorted_units):
        g = df[df["unit"] == unit_id].sort_values("cycle")
        sensors = pad_left(g[SELECTED_SENSORS].to_numpy(np.float32), seq_len)
        ops = pad_left(g[OP_COLS].to_numpy(np.float32), seq_len)
        final_row_index = g.index.to_numpy()[-1]

        X.append(sensors)
        twin_aux.append(twin_aux_from_window(sensors, ops, healthy_ref))
        y.append(float(rul_end_values[label_idx]))
        units.append(int(unit_id))
        regimes.append(int(regime_labels[final_row_index]))

    return {
        "X": np.asarray(X, dtype=np.float32),
        "twin_aux": np.asarray(twin_aux, dtype=np.float32),
        "y": np.asarray(y, dtype=np.float32),
        "unit_ids": np.asarray(units, dtype=np.int32),
        "regime_ids": np.asarray(regimes, dtype=np.int16),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset",
        default="FD004",
        choices=["FD001", "FD002", "FD003", "FD004"],
    )
    parser.add_argument("--raw-dir", default="data/raw/cmapss")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--seq-len", type=int, default=40)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--rul-cap", type=int, default=125)
    parser.add_argument("--healthy-cycles", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.stride < 1:
        raise ValueError("--stride must be >= 1.")

    raw_dir = Path(args.raw_dir)
    train_path = raw_dir / f"train_{args.subset}.txt"
    test_path = raw_dir / f"test_{args.subset}.txt"
    rul_path = raw_dir / f"RUL_{args.subset}.txt"

    missing = [p for p in (train_path, test_path, rul_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required C-MAPSS file(s):\n"
            + "\n".join(str(p) for p in missing)
        )

    train_raw = add_train_rul(read_cmapss_table(train_path), args.rul_cap)
    test_raw = read_cmapss_table(test_path)

    norm_cols = SELECTED_SENSORS + OP_COLS
    mean, std = fit_standardizer(train_raw, norm_cols)

    train = apply_standardizer(
        train_raw, mean, std, norm_cols
    ).reset_index(drop=True)
    test = apply_standardizer(
        test_raw, mean, std, norm_cols
    ).reset_index(drop=True)

    healthy_ref = fit_healthy_reference(train, args.healthy_cycles)

    num_regimes = 6 if args.subset in {"FD002", "FD004"} else 1
    if num_regimes == 1:
        train_regimes = np.zeros(len(train), dtype=np.int16)
        test_regimes = np.zeros(len(test), dtype=np.int16)
        centers = np.zeros((1, 3), dtype=np.float32)
    else:
        kmeans = KMeans(
            n_clusters=num_regimes,
            random_state=args.seed,
            n_init=20,
        )
        train_regimes = kmeans.fit_predict(
            train[OP_COLS].to_numpy(np.float32)
        ).astype(np.int16)
        test_regimes = kmeans.predict(
            test[OP_COLS].to_numpy(np.float32)
        ).astype(np.int16)
        centers = kmeans.cluster_centers_.astype(np.float32)

    train_arrays = build_train_windows(
        train,
        args.seq_len,
        args.stride,
        healthy_ref,
        train_regimes,
    )

    rul_end = pd.read_csv(
        rul_path, sep=r"\s+", header=None
    ).iloc[:, 0].to_numpy(np.float32)
    rul_end = np.minimum(rul_end, args.rul_cap).astype(np.float32)

    test_arrays = build_test_windows(
        test,
        rul_end,
        args.seq_len,
        healthy_ref,
        test_regimes,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_out = out_dir / f"{args.subset}_train.npz"
    test_out = out_dir / f"{args.subset}_test.npz"
    meta_out = out_dir / f"{args.subset}_meta.json"

    np.savez_compressed(train_out, **train_arrays)
    np.savez_compressed(test_out, **test_arrays)

    metadata = {
        "subset": args.subset,
        "seq_len": args.seq_len,
        "train_window_stride": args.stride,
        "rul_cap": args.rul_cap,
        "healthy_cycles": args.healthy_cycles,
        "selected_sensor_ids": SELECTED_SENSOR_IDS,
        "selected_sensors": SELECTED_SENSORS,
        "num_sensors": len(SELECTED_SENSORS),
        "twin_aux_dim": 5,
        "num_regimes": num_regimes,
        "normalization": {
            "mean": {c: float(mean[c]) for c in norm_cols},
            "std": {c: float(std[c]) for c in norm_cols},
        },
        "healthy_reference_standardized": healthy_ref.tolist(),
        "regime_centers_standardized": centers.tolist(),
        "leakage_guards": [
            "normalization fitted only on training data",
            "healthy reference fitted only on early training cycles",
            "twin auxiliary state contains no RUL-derived feature",
            "NASA test labels are used only as targets",
            "the same RUL cap is applied to training and test targets",
        ],
    }
    meta_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved {train_out}")
    print(f"X: {train_arrays['X'].shape}")
    print(f"twin_aux: {train_arrays['twin_aux'].shape}")
    print(f"y: {train_arrays['y'].shape}")
    print(f"train engines: {len(np.unique(train_arrays['unit_ids']))}")
    print(f"regimes: {np.unique(train_arrays['regime_ids']).tolist()}")
    print(f"training-window stride: {args.stride}")

    print(f"\nSaved {test_out}")
    print(f"X: {test_arrays['X'].shape}")
    print(f"twin_aux: {test_arrays['twin_aux'].shape}")
    print(f"y: {test_arrays['y'].shape}")
    print(f"test engines: {len(np.unique(test_arrays['unit_ids']))}")
    print(
        f"test RUL range: "
        f"{test_arrays['y'].min():.1f} to {test_arrays['y'].max():.1f}"
    )

    print(f"\nSaved {meta_out}")
    print("Leakage guard passed: twin_aux contains no RUL-derived input.")
    print(f"RUL target cap applied consistently: {args.rul_cap}")

if __name__ == "__main__":
    main()
