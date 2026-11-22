import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED = [
    "fedper_residual_v2",
    "twinfed_moe_v2",
    "v2_no_dt",
    "v2_no_moe",
    "v2_no_personalization",
    "v2_no_tgc",
    "v2_fedavg_aggregation",
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
        / "latest_v2.txt"
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
        / "v2_results.csv"
    )

    df = pd.read_csv(
        path
    )

    missing = [
        x
        for x in EXPECTED
        if x not in set(
            df["method"]
        )
    ]

    problems = []

    if missing:
        problems.append(
            f"Missing methods: {missing}"
        )

    required = [
        "shared_val_rmse",
        "personal_val_rmse",
        "test_rmse",
        "test_mae",
        "test_r2",
        "test_nasa",
        "train_seconds",
    ]

    for col in required:
        if col not in df.columns:
            problems.append(
                f"Missing column: {col}"
            )
            continue

        values = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        bad = ~np.isfinite(
            values
        )

        if bad.any():
            problems.append(
                f"Non-finite {col}: "
                f"{df.loc[bad, 'method'].tolist()}"
            )

    no_personal = df[
        df["method"]
        == "v2_no_personalization"
    ]

    if len(no_personal):
        delta = abs(
            float(
                no_personal.iloc[0][
                    "personalization_gain_rmse"
                ]
            )
        )

        if delta > 1e-5:
            problems.append(
                "No-personalization ablation has "
                "nonzero personalization gain."
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
            "\nCHECK FAILED"
        )

        for problem in problems:
            print(
                f" - {problem}"
            )

        raise SystemExit(
            1
        )

    print(
        "V2 integrity check passed."
    )


if __name__ == "__main__":
    main()
