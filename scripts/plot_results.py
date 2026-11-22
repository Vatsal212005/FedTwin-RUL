import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def resolve_run_dir(
    outputs_root,
    run_dir_arg,
):
    if run_dir_arg:
        p = Path(
            run_dir_arg
        )

        if not p.exists():
            raise FileNotFoundError(
                p
            )

        return p

    pointer = (
        Path(outputs_root)
        / "latest_run.txt"
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


def save_line(
    x,
    ys,
    labels,
    xlabel,
    ylabel,
    title,
    output_path,
):
    plt.figure(
        figsize=(7.2, 4.8)
    )

    for y, label in zip(
        ys,
        labels,
    ):
        plt.plot(
            x,
            y,
            label=label,
        )

    plt.xlabel(
        xlabel
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.grid(
        alpha=0.25
    )

    if len(labels) > 1:
        plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir",
        default=None,
    )

    parser.add_argument(
        "--outputs-root",
        default="outputs",
    )

    args = parser.parse_args()

    run_dir = resolve_run_dir(
        args.outputs_root,
        args.run_dir,
    )

    metrics_path = (
        run_dir
        / "round_metrics.csv"
    )

    df = pd.read_csv(
        metrics_path
    )

    fig_dir = (
        run_dir
        / "figures"
    )

    fig_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rounds = df[
        "round"
    ].to_numpy()

    save_line(
        rounds,
        [
            df["pooled_rmse"],
            df["shared_pooled_rmse"],
        ],
        [
            "Personalized TwinFed-MoE",
            "Shared-only TwinFed-MoE",
        ],
        "Federated round",
        "RMSE (cycles)",
        "Validation RMSE Convergence",
        fig_dir / "validation_rmse.png",
    )

    save_line(
        rounds,
        [
            df["pooled_mae"],
            df["shared_pooled_mae"],
        ],
        [
            "Personalized TwinFed-MoE",
            "Shared-only TwinFed-MoE",
        ],
        "Federated round",
        "MAE (cycles)",
        "Validation MAE Convergence",
        fig_dir / "validation_mae.png",
    )

    save_line(
        rounds,
        [
            df["pooled_r2"],
            df["shared_pooled_r2"],
        ],
        [
            "Personalized TwinFed-MoE",
            "Shared-only TwinFed-MoE",
        ],
        "Federated round",
        "R²",
        "Validation R² Convergence",
        fig_dir / "validation_r2.png",
    )

    save_line(
        rounds,
        [
            df["mean_train_loss"],
        ],
        [
            "Training objective",
        ],
        "Federated round",
        "Normalized training loss",
        "Federated Training Objective",
        fig_dir / "training_loss.png",
    )

    expert_cols = sorted(
        [
            c
            for c in df.columns
            if c.startswith(
                "expert_util_"
            )
            and c != (
                "expert_utilization_entropy"
            )
        ],
        key=lambda x: int(
            x.split("_")[-1]
        ),
    )

    if expert_cols:
        save_line(
            rounds,
            [
                df[c]
                for c in expert_cols
            ],
            [
                f"Expert {i}"
                for i in range(
                    len(expert_cols)
                )
            ],
            "Federated round",
            "Weighted utilization",
            "Expert Utilization Across Training",
            fig_dir / "expert_utilization.png",
        )

    predictions_path = (
        run_dir
        / "official_test_predictions.csv"
    )

    if predictions_path.exists():
        pred_df = pd.read_csv(
            predictions_path
        )

        plt.figure(
            figsize=(6.0, 6.0)
        )

        plt.scatter(
            pred_df["true_rul"],
            pred_df["pred_rul"],
            alpha=0.65,
        )

        lo = float(
            min(
                pred_df["true_rul"].min(),
                pred_df["pred_rul"].min(),
            )
        )

        hi = float(
            max(
                pred_df["true_rul"].max(),
                pred_df["pred_rul"].max(),
            )
        )

        plt.plot(
            [lo, hi],
            [lo, hi],
            linestyle="--",
            label="Ideal prediction",
        )

        plt.xlabel(
            "True RUL (cycles)"
        )

        plt.ylabel(
            "Predicted RUL (cycles)"
        )

        plt.title(
            "Official FD004 Test Predictions"
        )

        plt.legend()

        plt.grid(
            alpha=0.25
        )

        plt.tight_layout()

        plt.savefig(
            fig_dir
            / "official_predicted_vs_true.png",
            dpi=220,
            bbox_inches="tight",
        )

        plt.close()

        abs_error_sorted = np.sort(
            pred_df[
                "abs_error"
            ].to_numpy()
        )

        cdf = (
            np.arange(
                1,
                len(abs_error_sorted) + 1
            )
            / len(abs_error_sorted)
        )

        plt.figure(
            figsize=(7.2, 4.8)
        )

        plt.plot(
            abs_error_sorted,
            cdf,
        )

        plt.xlabel(
            "Absolute RUL error (cycles)"
        )

        plt.ylabel(
            "Empirical CDF"
        )

        plt.title(
            "Official FD004 Absolute-Error Distribution"
        )

        plt.grid(
            alpha=0.25
        )

        plt.tight_layout()

        plt.savefig(
            fig_dir
            / "official_abs_error_cdf.png",
            dpi=220,
            bbox_inches="tight",
        )

        plt.close()

    regime_path = (
        run_dir
        / "regime_expert_summary.csv"
    )

    if regime_path.exists():
        regime_df = pd.read_csv(
            regime_path
        )

        expert_cols = [
            c
            for c in regime_df.columns
            if c.startswith(
                "expert_util_"
            )
        ]

        if expert_cols:
            x = np.arange(
                len(regime_df)
            )

            bottom = np.zeros(
                len(regime_df)
            )

            plt.figure(
                figsize=(7.5, 4.8)
            )

            for i, col in enumerate(
                expert_cols
            ):
                values = (
                    regime_df[col]
                    .to_numpy()
                )

                plt.bar(
                    x,
                    values,
                    bottom=bottom,
                    label=f"Expert {i}",
                )

                bottom += values

            plt.xticks(
                x,
                regime_df[
                    "regime_id"
                ].astype(str),
            )

            plt.xlabel(
                "Operating regime"
            )

            plt.ylabel(
                "Mean sparse expert utilization"
            )

            plt.title(
                "Digital-Twin-Guided Expert Specialization"
            )

            plt.legend()

            plt.tight_layout()

            plt.savefig(
                fig_dir
                / "regime_expert_utilization.png",
                dpi=220,
                bbox_inches="tight",
            )

            plt.close()

    print(
        f"Saved figures to "
        f"{fig_dir}"
    )


if __name__ == "__main__":
    main()
