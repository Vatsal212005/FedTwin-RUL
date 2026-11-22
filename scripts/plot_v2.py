import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = {
    "fedper_residual_v2": "FedPer\nResidual v2",
    "twinfed_moe_v2": "TwinFed-MoE\nv2",
    "v2_no_dt": "No DT",
    "v2_no_moe": "No MoE",
    "v2_no_personalization": "No Personal.",
    "v2_no_tgc": "No TGC",
    "v2_fedavg_aggregation": "FedAvg Agg.",
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

    df = pd.read_csv(
        suite
        / "v2_results.csv"
    )

    fig_dir = (
        suite
        / "figures"
    )

    fig_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels = [
        LABELS.get(
            m,
            m,
        )
        for m in df[
            "method"
        ]
    ]

    x = np.arange(
        len(df)
    )

    plt.figure(
        figsize=(10.5, 5.2)
    )

    width = 0.36

    plt.bar(
        x - width / 2,
        df[
            "shared_val_rmse"
        ],
        width=width,
        label="Shared validation",
    )

    plt.bar(
        x + width / 2,
        df[
            "personal_val_rmse"
        ],
        width=width,
        label="Personalized validation",
    )

    plt.xticks(
        x,
        labels,
        rotation=30,
        ha="right",
    )

    plt.ylabel(
        "RMSE (cycles)"
    )

    plt.title(
        "V2 Shared vs Personalized Validation"
    )

    plt.legend()
    plt.grid(
        axis="y",
        alpha=0.25,
    )
    plt.tight_layout()

    plt.savefig(
        fig_dir
        / "v2_shared_vs_personalized.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close()

    plt.figure(
        figsize=(10.5, 5.2)
    )

    plt.bar(
        x,
        df[
            "test_rmse"
        ],
    )

    plt.xticks(
        x,
        labels,
        rotation=30,
        ha="right",
    )

    plt.ylabel(
        "RMSE (cycles)"
    )

    plt.title(
        "V2 Official FD004 Test RMSE"
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )
    plt.tight_layout()

    plt.savefig(
        fig_dir
        / "v2_test_rmse.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close()

    proposed_dir = (
        suite
        / "twinfed_moe_v2"
    )

    round_path = (
        proposed_dir
        / "round_metrics.csv"
    )

    if round_path.exists():
        r = pd.read_csv(
            round_path
        )

        plt.figure(
            figsize=(7.2, 4.8)
        )

        plt.plot(
            r["round"],
            r[
                "shared_pooled_rmse"
            ],
            label="Shared",
        )

        plt.plot(
            r["round"],
            r[
                "personal_pooled_rmse"
            ],
            label="Personalized",
        )

        plt.xlabel(
            "Federated round"
        )
        plt.ylabel(
            "RMSE (cycles)"
        )
        plt.title(
            "TwinFed-MoE v2 Convergence"
        )
        plt.legend()
        plt.grid(
            alpha=0.25,
        )
        plt.tight_layout()

        plt.savefig(
            fig_dir
            / "twinfed_v2_convergence.png",
            dpi=220,
            bbox_inches="tight",
        )

        plt.close()

    print(
        f"Saved V2 figures to "
        f"{fig_dir}"
    )


if __name__ == "__main__":
    main()
