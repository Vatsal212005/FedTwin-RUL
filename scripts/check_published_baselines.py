import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_METHODS = {
    "trans_lighter_repro",
    "scaffold_transformer",
}

EXPECTED_ALPHAS = {
    1.0,
    0.5,
    0.1,
}

EXPECTED_SEEDS = {
    42,
    123,
    2027,
}


def resolve_study(
    study_dir,
    outputs_root,
):
    if study_dir:
        return Path(
            study_dir
        )

    pointer = (
        Path(
            outputs_root
        )
        / "latest_published_baselines.txt"
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
        "--study-dir",
        default=None,
    )

    parser.add_argument(
        "--outputs-root",
        default="outputs",
    )

    args = parser.parse_args()

    study = resolve_study(
        args.study_dir,
        args.outputs_root,
    )

    raw = pd.read_csv(
        study
        / "published_raw_results.csv"
    )

    expected_total = (
        len(
            EXPECTED_METHODS
        )
        * len(
            EXPECTED_ALPHAS
        )
        * len(
            EXPECTED_SEEDS
        )
    )

    problems = []

    if len(
        raw
    ) != expected_total:
        problems.append(
            f"Expected "
            f"{expected_total} runs, "
            f"found {len(raw)}."
        )

    combos = set(
        zip(
            raw[
                "method"
            ].astype(
                str
            ),
            raw[
                "alpha"
            ].astype(
                float
            ),
            raw[
                "seed"
            ].astype(
                int
            ),
        )
    )

    for method in (
        EXPECTED_METHODS
    ):
        for alpha in (
            EXPECTED_ALPHAS
        ):
            for seed in (
                EXPECTED_SEEDS
            ):
                if (
                    method,
                    alpha,
                    seed,
                ) not in combos:
                    problems.append(
                        "Missing "
                        f"{method}, "
                        f"alpha={alpha}, "
                        f"seed={seed}"
                    )

    if raw.duplicated(
        subset=[
            "method",
            "alpha",
            "seed",
        ]
    ).any():
        problems.append(
            "Duplicate "
            "method/alpha/seed "
            "rows found."
        )

    for col in [
        "val_rmse",
        "val_mae",
        "val_r2",
        "test_rmse",
        "test_mae",
        "test_r2",
        "test_nasa",
        "train_seconds",
        "parameters",
    ]:
        values = pd.to_numeric(
            raw[
                col
            ],
            errors="coerce",
        )

        if (
            ~np.isfinite(
                values
            )
        ).any():
            problems.append(
                f"Non-finite "
                f"values in {col}."
            )

    print(
        f"Study: {study}"
    )

    print(
        f"Completed runs: "
        f"{len(raw)} / "
        f"{expected_total}"
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
        "Published-baseline "
        "integrity check passed."
    )


if __name__ == "__main__":
    main()
