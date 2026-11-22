import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_METHODS = {
    "fedavg_transformer",
    "fedprox_transformer",
    "fedavg_moe",
    "fedtwin_rul",
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


def resolve_study(study_dir, outputs_root):
    if study_dir:
        return Path(study_dir)

    pointer = Path(outputs_root) / "latest_fedtwin_final.txt"

    if not pointer.exists():
        raise FileNotFoundError(pointer)

    return Path(
        pointer.read_text(
            encoding="utf-8"
        ).strip()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", default=None)
    parser.add_argument("--outputs-root", default="outputs")
    args = parser.parse_args()

    study = resolve_study(
        args.study_dir,
        args.outputs_root,
    )

    raw_path = study / "raw_results.csv"
    summary_path = study / "mean_std_results.csv"

    raw = pd.read_csv(raw_path)
    summary = pd.read_csv(summary_path)

    problems = []

    expected_total = (
        len(EXPECTED_METHODS)
        * len(EXPECTED_ALPHAS)
        * len(EXPECTED_SEEDS)
    )

    if len(raw) != expected_total:
        problems.append(
            f"Expected {expected_total} runs, found {len(raw)}."
        )

    combinations = set(
        zip(
            raw["method"].astype(str),
            raw["alpha"].astype(float),
            raw["seed"].astype(int),
        )
    )

    for method in EXPECTED_METHODS:
        for alpha in EXPECTED_ALPHAS:
            for seed in EXPECTED_SEEDS:
                if (
                    method,
                    alpha,
                    seed,
                ) not in combinations:
                    problems.append(
                        f"Missing {method}, alpha={alpha}, seed={seed}"
                    )

    if raw.duplicated(
        subset=[
            "method",
            "alpha",
            "seed",
        ]
    ).any():
        problems.append(
            "Duplicate method/alpha/seed runs found."
        )

    required_numeric = [
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

    for col in required_numeric:
        values = pd.to_numeric(
            raw[col],
            errors="coerce",
        )

        bad = ~np.isfinite(values)

        if bad.any():
            problems.append(
                f"Non-finite {col}: "
                f"{raw.loc[bad, ['method','alpha','seed']].to_dict('records')}"
            )

    if len(summary) != (
        len(EXPECTED_METHODS)
        * len(EXPECTED_ALPHAS)
    ):
        problems.append(
            "Mean/std table has incorrect number of rows."
        )

    print(f"Study: {study}")
    print(
        f"Completed runs: {len(raw)} / {expected_total}"
    )

    if problems:
        print("\nCHECK FAILED")

        for problem in problems:
            print(f" - {problem}")

        raise SystemExit(1)

    print(
        "Final FedTwin-RUL study integrity check passed."
    )


if __name__ == "__main__":
    main()
