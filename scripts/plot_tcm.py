import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = {
    "dt_concat_fedavg": "DT Concat\n+ FedAvg",
    "tcm_fedavg": "TCM\n+ FedAvg",
    "tcm_fedprox": "TCM\n+ FedProx",
    "tcm_no_dt_reconstruction": "TCM\nNo Recon",
    "tcm_identity_modulation": "Identity\nControl",
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
        / "tcm_results.csv"
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
            method,
            method,
        )
        for method in (
            df[
                "method"
            ].tolist()
        )
    ]

    x = np.arange(
        len(df)
    )

    plt.figure(
        figsize=(9.5, 5.0)
    )

    width = 0.36

    plt.bar(
        x - width / 2,
        df[
            "val_rmse"
        ],
        width=width,
        label="Validation",
    )

    plt.bar(
        x + width / 2,
        df[
            "test_rmse"
        ],
        width=width,
        label="Official FD004 test",
    )

    plt.xticks(
        x,
        labels,
    )

    plt.ylabel(
        "RMSE (cycles)"
    )

    plt.title(
        "TwinFed-TCM Controlled Comparison"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        fig_dir
        / "tcm_val_test_rmse.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close()

    tcm_rows = df[
        df[
            "fusion_mode"
        ] == "tcm"
    ]

    if len(tcm_rows):
        x2 = np.arange(
            len(
                tcm_rows
            )
        )

        labels2 = [
            LABELS.get(
                method,
                method,
            )
            for method in (
                tcm_rows[
                    "method"
                ].tolist()
            )
        ]

        plt.figure(
            figsize=(8.6, 4.8)
        )

        plt.bar(
            x2,
            tcm_rows[
                "gamma_abs_shift_mean"
            ],
        )

        plt.xticks(
            x2,
            labels2,
        )

        plt.ylabel(
            r"Mean $|\gamma-1|$"
        )

        plt.title(
            "Magnitude of Learned Twin Conditioning"
        )

        plt.grid(
            axis="y",
            alpha=0.25,
        )

        plt.tight_layout()

        plt.savefig(
            fig_dir
            / "tcm_gamma_modulation.png",
            dpi=220,
            bbox_inches="tight",
        )

        plt.close()

    proposed_path = (
        suite
        / "tcm_fedprox"
        / "round_metrics.csv"
    )

    if proposed_path.exists():
        rounds = pd.read_csv(
            proposed_path
        )

        plt.figure(
            figsize=(7.2, 4.8)
        )

        plt.plot(
            rounds[
                "round"
            ],
            rounds[
                "pooled_rmse"
            ],
        )

        plt.xlabel(
            "Federated round"
        )

        plt.ylabel(
            "Validation RMSE (cycles)"
        )

        plt.title(
            "TwinFed-TCM Convergence"
        )

        plt.grid(
            alpha=0.25,
        )

        plt.tight_layout()

        plt.savefig(
            fig_dir
            / "tcm_fedprox_convergence.png",
            dpi=220,
            bbox_inches="tight",
        )

        plt.close()

    regime_path = (
        suite
        / "tcm_fedprox"
        / "regime_diagnostics.csv"
    )

    if regime_path.exists():
        regime = pd.read_csv(
            regime_path
        )

        plt.figure(
            figsize=(7.2, 4.8)
        )

        plt.bar(
            regime[
                "regime_id"
            ].astype(
                str
            ),
            regime[
                "rmse"
            ],
        )

        plt.xlabel(
            "Operating regime"
        )

        plt.ylabel(
            "RMSE (cycles)"
        )

        plt.title(
            "TwinFed-TCM Performance by Operating Regime"
        )

        plt.grid(
            axis="y",
            alpha=0.25,
        )

        plt.tight_layout()

        plt.savefig(
            fig_dir
            / "tcm_regime_rmse.png",
            dpi=220,
            bbox_inches="tight",
        )

        plt.close()

    print(
        f"Saved TCM figures to "
        f"{fig_dir}"
    )


if __name__ == "__main__":
    main()
