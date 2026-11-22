import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED = [
    "dt_concat_fedavg",
    "tcm_fedavg",
    "tcm_fedprox",
    "tcm_no_dt_reconstruction",
    "tcm_identity_modulation",
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
        / "latest_tcm.txt"
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
        / "tcm_results.csv"
    )

    df = pd.read_csv(
        path
    )

    problems = []

    found = set(
        df[
            "method"
        ].tolist()
    )

    missing = [
        method
        for method in EXPECTED
        if method not in found
    ]

    if missing:
        problems.append(
            f"Missing methods: "
            f"{missing}"
        )

    if (
        df[
            "method"
        ].duplicated()
        .any()
    ):
        problems.append(
            "Duplicate method rows found."
        )

    required = [
        "val_rmse",
        "val_mae",
        "val_r2",
        "test_rmse",
        "test_mae",
        "test_r2",
        "test_nasa",
        "train_seconds",
        "parameters",
    ]

    for col in required:
        if col not in df.columns:
            problems.append(
                f"Missing column: "
                f"{col}"
            )
            continue

        numeric = (
            pd.to_numeric(
                df[
                    col
                ],
                errors="coerce",
            )
        )

        bad = ~np.isfinite(
            numeric
        )

        if bad.any():
            problems.append(
                f"Non-finite "
                f"{col}: "
                f"{df.loc[bad, 'method'].tolist()}"
            )

    identity = df[
        df[
            "method"
        ] == (
            "tcm_identity_modulation"
        )
    ]

    if len(identity):
        row = identity.iloc[
            0
        ]

        if (
            abs(
                float(
                    row[
                        "gamma_abs_shift_mean"
                    ]
                )
            )
            > 1e-6
            or abs(
                float(
                    row[
                        "beta_abs_mean_mean"
                    ]
                )
            )
            > 1e-6
        ):
            problems.append(
                "Identity-control modulation "
                "diagnostics are not zero."
            )

    print(
        f"Suite: {suite}"
    )

    print(
        f"Completed methods: "
        f"{len(df)} / "
        f"{len(EXPECTED)}"
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
        "TCM integrity check passed."
    )


if __name__ == "__main__":
    main()
