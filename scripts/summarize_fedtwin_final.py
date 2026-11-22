import argparse
from pathlib import Path

import pandas as pd


LABELS = {
    "fedavg_transformer": "FedAvg Transformer",
    "fedprox_transformer": "FedProx Transformer",
    "fedavg_moe": "FedAvg-MoE",
    "fedtwin_rul": "FedTwin-RUL",
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

    raw = pd.read_csv(
        study / "raw_results.csv"
    )

    summary = pd.read_csv(
        study / "mean_std_results.csv"
    )

    print(f"Study: {study}")
    print(
        f"Completed runs: {len(raw)}"
    )

    print("\nVALIDATION RMSE BY HETEROGENEITY")

    for alpha in sorted(
        summary["alpha"].unique(),
        reverse=True,
    ):
        print(f"\n  alpha={alpha}")

        block = (
            summary[
                summary["alpha"] == alpha
            ]
            .sort_values(
                "val_rmse_mean"
            )
        )

        for _, row in block.iterrows():
            label = LABELS.get(
                row["method"],
                row["method"],
            )

            print(
                f"    {label:<24} "
                f"{row['val_rmse_mean']:.3f} "
                f"± {row['val_rmse_std']:.3f}"
            )

    print("\nOFFICIAL FD004 TEST RMSE")

    for alpha in sorted(
        summary["alpha"].unique(),
        reverse=True,
    ):
        print(f"\n  alpha={alpha}")

        block = (
            summary[
                summary["alpha"] == alpha
            ]
            .sort_values(
                "test_rmse_mean"
            )
        )

        for _, row in block.iterrows():
            label = LABELS.get(
                row["method"],
                row["method"],
            )

            print(
                f"    {label:<24} "
                f"{row['test_rmse_mean']:.3f} "
                f"± {row['test_rmse_std']:.3f}"
            )

    print("\nFEDTWIN-RUL VS FEDAVG")

    for alpha in sorted(
        summary["alpha"].unique(),
        reverse=True,
    ):
        ft = summary[
            (summary["alpha"] == alpha)
            & (
                summary["method"]
                == "fedtwin_rul"
            )
        ]

        fa = summary[
            (summary["alpha"] == alpha)
            & (
                summary["method"]
                == "fedavg_transformer"
            )
        ]

        if not len(ft) or not len(fa):
            continue

        ft = ft.iloc[0]
        fa = fa.iloc[0]

        val_improvement = (
            (
                fa["val_rmse_mean"]
                - ft["val_rmse_mean"]
            )
            / fa["val_rmse_mean"]
            * 100.0
        )

        test_improvement = (
            (
                fa["test_rmse_mean"]
                - ft["test_rmse_mean"]
            )
            / fa["test_rmse_mean"]
            * 100.0
        )

        print(
            f"  alpha={alpha}: "
            f"validation {val_improvement:+.2f}% | "
            f"test {test_improvement:+.2f}%"
        )

    print("\nEFFICIENCY")

    efficiency = (
        summary.groupby(
            "method"
        )[
            [
                "train_seconds_mean",
                "parameters_mean",
            ]
        ]
        .mean()
        .reset_index()
    )

    for _, row in (
        efficiency.sort_values(
            "parameters_mean"
        ).iterrows()
    ):
        label = LABELS.get(
            row["method"],
            row["method"],
        )

        print(
            f"  {label:<24} "
            f"params={int(row['parameters_mean']):,} | "
            f"time={row['train_seconds_mean']:.1f}s"
        )


if __name__ == "__main__":
    main()
