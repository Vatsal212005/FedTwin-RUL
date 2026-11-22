import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = {
    "centralized_transformer": "Centralized\nTransformer",
    "local_only_transformer": "Local-only\nTransformer",
    "fedavg_transformer": "FedAvg\nTransformer",
    "fedprox_transformer": "FedProx\nTransformer",
    "fedper_transformer": "FedPer\nTransformer",
    "fedavg_moe": "FedAvg-MoE",
    "twinfed_moe": "TwinFed-MoE",
    "ablation_no_dt": "No DT",
    "ablation_no_moe": "No MoE",
    "ablation_no_personalization": "No Personal.",
    "ablation_no_tgc": "No TGC",
    "ablation_fedavg_aggregation": "FedAvg Agg.",
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
        / "latest_benchmark.txt"
    )

    return Path(
        pointer.read_text(
            encoding="utf-8"
        ).strip()
    )


def bar_plot(
    df,
    value_col,
    title,
    ylabel,
    output_path,
):
    plot_df = (
        df.dropna(
            subset=[
                value_col
            ]
        )
        .copy()
    )

    labels = [
        LABELS.get(
            x,
            x,
        )
        for x in (
            plot_df[
                "method"
            ].tolist()
        )
    ]

    values = (
        plot_df[
            value_col
        ].to_numpy()
    )

    x = np.arange(
        len(values)
    )

    plt.figure(
        figsize=(11.0, 5.4)
    )

    plt.bar(
        x,
        values,
    )

    plt.xticks(
        x,
        labels,
        rotation=35,
        ha="right",
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

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
        / "benchmark_results.csv"
    )

    fig_dir = (
        suite
        / "figures"
    )

    fig_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    main_methods = df[
        df["category"].isin(
            [
                "reference",
                "baseline",
                "proposed",
            ]
        )
    ]

    ablations = df[
        df["category"].isin(
            [
                "proposed",
                "ablation",
            ]
        )
    ]

    bar_plot(
        main_methods,
        "test_rmse",
        "Official FD004 Test RMSE: Main Comparison",
        "RMSE (cycles)",
        fig_dir
        / "main_test_rmse.png",
    )

    bar_plot(
        main_methods,
        "test_mae",
        "Official FD004 Test MAE: Main Comparison",
        "MAE (cycles)",
        fig_dir
        / "main_test_mae.png",
    )

    bar_plot(
        main_methods,
        "test_nasa",
        "Official FD004 NASA Score: Main Comparison",
        "NASA asymmetric score",
        fig_dir
        / "main_test_nasa.png",
    )

    bar_plot(
        ablations,
        "test_rmse",
        "TwinFed-MoE Ablation Study: Official Test RMSE",
        "RMSE (cycles)",
        fig_dir
        / "ablation_test_rmse.png",
    )

    bar_plot(
        ablations,
        "val_rmse",
        "TwinFed-MoE Ablation Study: Validation RMSE",
        "RMSE (cycles)",
        fig_dir
        / "ablation_validation_rmse.png",
    )

    bar_plot(
        df,
        "train_seconds",
        "Training Runtime by Method",
        "Training time (seconds)",
        fig_dir
        / "runtime_comparison.png",
    )

    print(
        f"Saved benchmark figures to "
        f"{fig_dir}"
    )


if __name__ == "__main__":
    main()
