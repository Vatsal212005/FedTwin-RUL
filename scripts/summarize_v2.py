import argparse
from pathlib import Path

import pandas as pd


LABELS = {
    "centralized_transformer": "Centralized Transformer",
    "local_only_transformer": "Local-only Transformer",
    "fedavg_transformer": "FedAvg Transformer",
    "fedprox_transformer": "FedProx Transformer",
    "fedper_transformer": "FedPer Transformer (v1)",
    "fedavg_moe": "FedAvg-MoE",
    "twinfed_moe": "TwinFed-MoE (v1)",
    "fedper_residual_v2": "FedPer Residual v2",
    "twinfed_moe_v2": "TwinFed-MoE v2",
    "v2_no_dt": "V2 w/o DT",
    "v2_no_moe": "V2 w/o MoE",
    "v2_no_personalization": "V2 w/o Personalization",
    "v2_no_tgc": "V2 w/o TGC",
    "v2_fedavg_aggregation": "V2 + FedAvg Aggregation",
}


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

    v2_path = (
        suite
        / "v2_results.csv"
    )

    if not v2_path.exists():
        raise FileNotFoundError(
            v2_path
        )

    v2 = pd.read_csv(
        v2_path
    )

    v2["label"] = (
        v2["method"]
        .map(LABELS)
        .fillna(
            v2["method"]
        )
    )

    print(
        f"Suite: {suite}"
    )

    print(
        "\nV2 SHARED-VALIDATION RANKING"
    )

    for _, row in (
        v2.sort_values(
            "shared_val_rmse"
        ).iterrows()
    ):
        print(
            f"  {row['label']:<32} "
            f"shared={row['shared_val_rmse']:.4f}  "
            f"personal={row['personal_val_rmse']:.4f}  "
            f"gain={row['personalization_gain_rmse']:+.4f}"
        )

    print(
        "\nV2 OFFICIAL FD004 RANKING"
    )

    for _, row in (
        v2.sort_values(
            "test_rmse"
        ).iterrows()
    ):
        print(
            f"  {row['label']:<32} "
            f"RMSE={row['test_rmse']:.4f}  "
            f"MAE={row['test_mae']:.4f}  "
            f"R2={row['test_r2']:.4f}  "
            f"NASA={row['test_nasa']:.2f}"
        )

    proposed_rows = v2[
        v2["method"]
        == "twinfed_moe_v2"
    ]

    if len(proposed_rows):
        p = proposed_rows.iloc[0]

        print(
            "\nTWINFED-MOE V2"
        )

        print(
            f"  best round: "
            f"{int(p['best_round'])}"
        )

        print(
            f"  shared validation RMSE: "
            f"{p['shared_val_rmse']:.4f}"
        )

        print(
            f"  personalized validation RMSE: "
            f"{p['personal_val_rmse']:.4f}"
        )

        print(
            f"  personalization gain: "
            f"{p['personalization_gain_rmse']:+.4f}"
        )

        print(
            f"  mean |residual|: "
            f"{p['residual_abs_mean_cycles']:.3f} cycles"
        )

        print(
            f"  p95 |residual|: "
            f"{p['residual_abs_p95_cycles']:.3f} cycles"
        )

        print(
            f"  official test RMSE: "
            f"{p['test_rmse']:.4f}"
        )

        print(
            "\nV2 ABLATION DELTAS VS FULL V2"
        )

        ablations = v2[
            v2["category"]
            == "ablation_v2"
        ]

        for _, row in (
            ablations.iterrows()
        ):
            print(
                f"  {row['label']:<32} "
                f"ΔSharedVal="
                f"{row['shared_val_rmse'] - p['shared_val_rmse']:+.4f}  "
                f"ΔTest="
                f"{row['test_rmse'] - p['test_rmse']:+.4f}"
            )

    merged_path = (
        suite
        / "v2_vs_fixed_baselines.csv"
    )

    if merged_path.exists():
        merged = pd.read_csv(
            merged_path
        )

        merged["label"] = (
            merged["method"]
            .map(LABELS)
            .fillna(
                merged["method"]
            )
        )

        print(
            "\nMERGED OFFICIAL-TEST RANKING"
        )

        ranked = (
            merged.dropna(
                subset=[
                    "test_rmse"
                ]
            )
            .sort_values(
                "test_rmse"
            )
        )

        for _, row in (
            ranked.iterrows()
        ):
            print(
                f"  {row['label']:<32} "
                f"RMSE={row['test_rmse']:.4f}"
            )


if __name__ == "__main__":
    main()
