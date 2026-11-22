import argparse
from pathlib import Path

import pandas as pd


LABELS = {
    "centralized_transformer": "Centralized Transformer",
    "local_only_transformer": "Local-only Transformer",
    "fedavg_transformer": "FedAvg Transformer",
    "fedprox_transformer": "FedProx Transformer",
    "fedavg_moe": "FedAvg-MoE",
    "dt_concat_fedavg": "DT Concatenation + FedAvg",
    "tcm_fedavg": "TwinFed-TCM + FedAvg",
    "tcm_fedprox": "TwinFed-TCM + FedProx",
    "tcm_no_dt_reconstruction": "TCM w/o DT Reconstruction",
    "tcm_identity_modulation": "TCM Identity / No Conditioning",
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

    tcm = pd.read_csv(
        suite
        / "tcm_results.csv"
    )

    tcm[
        "label"
    ] = (
        tcm[
            "method"
        ]
        .map(
            LABELS
        )
        .fillna(
            tcm[
                "method"
            ]
        )
    )

    print(
        f"Suite: {suite}"
    )

    print(
        "\nTCM VALIDATION RANKING"
    )

    for _, row in (
        tcm.sort_values(
            "val_rmse"
        ).iterrows()
    ):
        print(
            f"  {row['label']:<34} "
            f"RMSE={row['val_rmse']:.4f}  "
            f"MAE={row['val_mae']:.4f}  "
            f"R2={row['val_r2']:.4f}"
        )

    print(
        "\nTCM OFFICIAL FD004 RANKING"
    )

    for _, row in (
        tcm.sort_values(
            "test_rmse"
        ).iterrows()
    ):
        print(
            f"  {row['label']:<34} "
            f"RMSE={row['test_rmse']:.4f}  "
            f"MAE={row['test_mae']:.4f}  "
            f"R2={row['test_r2']:.4f}  "
            f"NASA={row['test_nasa']:.2f}"
        )

    by_method = {
        row[
            "method"
        ]: row
        for _, row in (
            tcm.iterrows()
        )
    }

    print(
        "\nMECHANISM CHECKS"
    )

    if (
        "tcm_fedavg"
        in by_method
        and "dt_concat_fedavg"
        in by_method
    ):
        a = by_method[
            "tcm_fedavg"
        ]
        b = by_method[
            "dt_concat_fedavg"
        ]

        print(
            "  TCM vs simple concatenation:"
        )

        print(
            f"    validation ΔRMSE "
            f"(TCM - concat) = "
            f"{a['val_rmse'] - b['val_rmse']:+.4f}"
        )

        print(
            f"    test ΔRMSE "
            f"(TCM - concat) = "
            f"{a['test_rmse'] - b['test_rmse']:+.4f}"
        )

    if (
        "tcm_fedprox"
        in by_method
        and "tcm_identity_modulation"
        in by_method
    ):
        a = by_method[
            "tcm_fedprox"
        ]
        b = by_method[
            "tcm_identity_modulation"
        ]

        print(
            "  TCM conditioning vs identity control:"
        )

        print(
            f"    validation ΔRMSE "
            f"(TCM - identity) = "
            f"{a['val_rmse'] - b['val_rmse']:+.4f}"
        )

        print(
            f"    test ΔRMSE "
            f"(TCM - identity) = "
            f"{a['test_rmse'] - b['test_rmse']:+.4f}"
        )

    if (
        "tcm_fedavg"
        in by_method
        and "tcm_fedprox"
        in by_method
    ):
        a = by_method[
            "tcm_fedavg"
        ]
        b = by_method[
            "tcm_fedprox"
        ]

        print(
            "  FL optimizer under identical TCM:"
        )

        print(
            f"    validation ΔRMSE "
            f"(FedProx - FedAvg) = "
            f"{b['val_rmse'] - a['val_rmse']:+.4f}"
        )

        print(
            f"    test ΔRMSE "
            f"(FedProx - FedAvg) = "
            f"{b['test_rmse'] - a['test_rmse']:+.4f}"
        )

    if (
        "tcm_fedprox"
        in by_method
        and "tcm_no_dt_reconstruction"
        in by_method
    ):
        a = by_method[
            "tcm_fedprox"
        ]
        b = by_method[
            "tcm_no_dt_reconstruction"
        ]

        print(
            "  DT reconstruction objective:"
        )

        print(
            f"    validation ΔRMSE "
            f"(no-recon - full) = "
            f"{b['val_rmse'] - a['val_rmse']:+.4f}"
        )

        print(
            f"    test ΔRMSE "
            f"(no-recon - full) = "
            f"{b['test_rmse'] - a['test_rmse']:+.4f}"
        )

    merged_path = (
        suite
        / "tcm_vs_fixed_baselines.csv"
    )

    if merged_path.exists():
        merged = pd.read_csv(
            merged_path
        )

        merged[
            "label"
        ] = (
            merged[
                "method"
            ]
            .map(
                LABELS
            )
            .fillna(
                merged[
                    "method"
                ]
            )
        )

        print(
            "\nMERGED TEST RANKING"
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
                f"  {row['label']:<34} "
                f"RMSE={row['test_rmse']:.4f}"
            )


if __name__ == "__main__":
    main()
