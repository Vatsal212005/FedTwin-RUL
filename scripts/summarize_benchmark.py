import argparse
from pathlib import Path

import pandas as pd


LABELS = {
    "centralized_transformer": "Centralized Transformer",
    "local_only_transformer": "Local-only Transformer",
    "fedavg_transformer": "FedAvg Transformer",
    "fedprox_transformer": "FedProx Transformer",
    "fedper_transformer": "FedPer Transformer",
    "fedavg_moe": "FedAvg-MoE",
    "twinfed_moe": "TwinFed-MoE",
    "ablation_no_dt": "TwinFed-MoE w/o DT",
    "ablation_no_moe": "TwinFed-MoE w/o MoE",
    "ablation_no_personalization": "TwinFed-MoE w/o Personalization",
    "ablation_no_tgc": "TwinFed-MoE w/o TGC",
    "ablation_fedavg_aggregation": "TwinFed-MoE + FedAvg Aggregation",
}


def resolve_suite(
    suite_dir,
    outputs_root,
):
    if suite_dir:
        p = Path(
            suite_dir
        )

        if not p.exists():
            raise FileNotFoundError(
                p
            )

        return p

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

    df = pd.read_csv(
        path
    )

    df["label"] = (
        df["method"]
        .map(LABELS)
        .fillna(
            df["method"]
        )
    )

    print(
        f"Suite: {suite}"
    )

    print(
        "\nVALIDATION RANKING"
    )

    ranked_val = (
        df.sort_values(
            "val_rmse"
        )
    )

    for _, row in (
        ranked_val.iterrows()
    ):
        print(
            f"  {row['label']:<36} "
            f"RMSE={row['val_rmse']:.4f}  "
            f"MAE={row['val_mae']:.4f}  "
            f"R2={row['val_r2']:.4f}"
        )

    print(
        "\nOFFICIAL FD004 TEST RANKING"
    )

    ranked_test = (
        df.dropna(
            subset=[
                "test_rmse"
            ]
        )
        .sort_values(
            "test_rmse"
        )
    )

    for _, row in (
        ranked_test.iterrows()
    ):
        print(
            f"  {row['label']:<36} "
            f"RMSE={row['test_rmse']:.4f}  "
            f"MAE={row['test_mae']:.4f}  "
            f"R2={row['test_r2']:.4f}  "
            f"NASA={row['test_nasa']:.2f}"
        )

    proposed_rows = df[
        df["method"]
        == "twinfed_moe"
    ]

    if len(proposed_rows):
        proposed = (
            proposed_rows.iloc[0]
        )

        print(
            "\nPROPOSED METHOD"
        )

        print(
            f"  validation RMSE: "
            f"{proposed['val_rmse']:.4f}"
        )

        print(
            f"  official test RMSE: "
            f"{proposed['test_rmse']:.4f}"
        )

        print(
            f"  official test MAE: "
            f"{proposed['test_mae']:.4f}"
        )

        print(
            f"  official test R2: "
            f"{proposed['test_r2']:.4f}"
        )

        print(
            f"  official NASA score: "
            f"{proposed['test_nasa']:.2f}"
        )

        print(
            f"  personalization gain "
            f"(validation RMSE): "
            f"{proposed['personalization_gain_rmse']:.4f}"
        )

        print(
            "\nABLATION DELTAS VS FULL MODEL"
        )

        ablations = df[
            df["category"]
            == "ablation"
        ]

        for _, row in (
            ablations.iterrows()
        ):
            val_delta = (
                row["val_rmse"]
                - proposed[
                    "val_rmse"
                ]
            )

            test_delta = (
                row["test_rmse"]
                - proposed[
                    "test_rmse"
                ]
            )

            print(
                f"  {row['label']:<36} "
                f"ΔValRMSE={val_delta:+.4f}  "
                f"ΔTestRMSE={test_delta:+.4f}"
            )

    print(
        "\nRUNTIME"
    )

    for _, row in (
        df.sort_values(
            "train_seconds"
        ).iterrows()
    ):
        print(
            f"  {row['label']:<36} "
            f"{row['train_seconds'] / 60.0:.2f} min"
        )


if __name__ == "__main__":
    main()
