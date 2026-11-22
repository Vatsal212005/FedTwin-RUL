import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED = [
    "centralized_transformer",
    "local_only_transformer",
    "fedavg_transformer",
    "fedprox_transformer",
    "fedper_transformer",
    "fedavg_moe",
    "twinfed_moe",
    "ablation_no_dt",
    "ablation_no_moe",
    "ablation_no_personalization",
    "ablation_no_tgc",
    "ablation_fedavg_aggregation",
]


def resolve_suite(
    suite_dir,
    outputs_root,
):
    if suite_dir:
        return Path(
            suite_dir
        )

    pointer = (
        Path(outputs_root)
        / "latest_benchmark.txt"
    )

    if not pointer.exists():
        raise FileNotFoundError(
            pointer
        )

    return Path(
        pointer.read_text(
            encoding="utf-8"
        ).strip()
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--suite-dir",
        default=None,
    )

    parser.add_argument(
        "--outputs-root",
        default="outputs",
    )

    args = parser.parse_args()

    suite = resolve_suite(
        args.suite_dir,
        args.outputs_root,
    )

    path = (
        suite
        / "benchmark_results.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    df = pd.read_csv(
        path
    )

    found = set(
        df["method"].tolist()
    )

    missing = [
        x
        for x in EXPECTED
        if x not in found
    ]

    duplicate_methods = (
        df["method"][
            df["method"].duplicated()
        ].tolist()
    )

    required_numeric = [
        "val_rmse",
        "val_mae",
        "val_r2",
        "train_seconds",
        "parameters",
    ]

    problems = []

    if missing:
        problems.append(
            f"Missing methods: {missing}"
        )

    if duplicate_methods:
        problems.append(
            f"Duplicate methods: "
            f"{duplicate_methods}"
        )

    for col in required_numeric:
        if col not in df.columns:
            problems.append(
                f"Missing column: {col}"
            )
            continue

        bad = ~np.isfinite(
            pd.to_numeric(
                df[col],
                errors="coerce",
            )
        )

        if bad.any():
            problems.append(
                f"Non-finite {col}: "
                f"{df.loc[bad, 'method'].tolist()}"
            )

    # All methods except local-only should have official test metrics.
    test_expected = df[
        df["method"]
        != "local_only_transformer"
    ]

    for col in [
        "test_rmse",
        "test_mae",
        "test_r2",
        "test_nasa",
    ]:
        bad = ~np.isfinite(
            pd.to_numeric(
                test_expected[col],
                errors="coerce",
            )
        )

        if bad.any():
            problems.append(
                f"Missing/non-finite "
                f"{col}: "
                f"{test_expected.loc[bad, 'method'].tolist()}"
            )

    print(
        f"Suite: {suite}"
    )

    print(
        f"Completed methods: "
        f"{len(df)} / {len(EXPECTED)}"
    )

    if problems:
        print(
            "\nCHECK FAILED:"
        )

        for problem in problems:
            print(
                f" - {problem}"
            )

        raise SystemExit(
            1
        )

    print(
        "Benchmark integrity check passed."
    )


if __name__ == "__main__":
    main()
