import argparse
from pathlib import Path

import pandas as pd


DISPLAY = {
    "fedavg_transformer": "FedAvg Transformer",
    "fedprox_transformer": "FedProx Transformer",
    "fedavg_moe": "FedAvg-MoE",
    "fedtwin_rul": "FedTwin-RUL",
    "trans_lighter_repro": "Trans-Lighter (protocol-aligned)",
    "scaffold_transformer": "SCAFFOLD Transformer",
}

ORIGIN = {
    "fedavg_transformer": (
        "Standard FedAvg; Transformer-RUL baseline family also evaluated "
        "by Kamei & Taghipour, RESS 233 (2023), 109130."
    ),
    "fedprox_transformer": (
        "FedProx (Li et al., MLSys 2020) with the common Transformer "
        "RUL backbone."
    ),
    "fedavg_moe": (
        "Internal capacity-control baseline; not presented as a "
        "published RUL method."
    ),
    "fedtwin_rul": "Proposed method.",
    "trans_lighter_repro": (
        "Nguyen et al., Computers in Industry 148 (2023), 103888; "
        "DOI 10.1016/j.compind.2023.103888."
    ),
    "scaffold_transformer": (
        "Karimireddy et al., ICML/PMLR 119 (2020), SCAFFOLD."
    ),
}

CATEGORY = {
    "fedavg_transformer": "published_standard_baseline",
    "fedprox_transformer": "published_standard_baseline",
    "fedavg_moe": "internal_capacity_control",
    "fedtwin_rul": "proposed",
    "trans_lighter_repro": "published_rul_specific_reproduction",
    "scaffold_transformer": "published_standard_baseline",
}


def read_pointer(
    outputs_root,
    name,
):
    path = (
        Path(
            outputs_root
        )
        / name
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    return Path(
        path.read_text(
            encoding="utf-8"
        ).strip()
    )


def fmt(
    mean,
    std,
    digits=3,
):
    return (
        f"{float(mean):.{digits}f} "
        f"$\\pm$ "
        f"{float(std):.{digits}f}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--final-study-dir",
        default=None,
    )

    parser.add_argument(
        "--published-study-dir",
        default=None,
    )

    parser.add_argument(
        "--outputs-root",
        default="outputs",
    )

    args = parser.parse_args()

    final_dir = (
        Path(
            args.final_study_dir
        )
        if args.final_study_dir
        else read_pointer(
            args.outputs_root,
            "latest_fedtwin_final.txt",
        )
    )

    published_dir = (
        Path(
            args.published_study_dir
        )
        if args.published_study_dir
        else read_pointer(
            args.outputs_root,
            "latest_published_baselines.txt",
        )
    )

    core = pd.read_csv(
        final_dir
        / "mean_std_results.csv"
    )

    published = pd.read_csv(
        published_dir
        / "published_mean_std.csv"
    )

    common_columns = [
        "method",
        "alpha",
        "n_seeds",
        "val_rmse_mean",
        "val_rmse_std",
        "val_mae_mean",
        "val_mae_std",
        "val_r2_mean",
        "val_r2_std",
        "mean_client_rmse_mean",
        "mean_client_rmse_std",
        "worst_client_rmse_mean",
        "worst_client_rmse_std",
        "test_rmse_mean",
        "test_rmse_std",
        "test_mae_mean",
        "test_mae_std",
        "test_r2_mean",
        "test_r2_std",
        "test_nasa_mean",
        "test_nasa_std",
        "train_seconds_mean",
        "train_seconds_std",
        "parameters_mean",
        "parameters_std",
    ]

    core = core[
        common_columns
    ].copy()

    published = published[
        common_columns
    ].copy()

    merged = pd.concat(
        [
            core,
            published,
        ],
        ignore_index=True,
    )

    merged[
        "display_name"
    ] = (
        merged[
            "method"
        ].map(
            DISPLAY
        )
    )

    merged[
        "origin"
    ] = (
        merged[
            "method"
        ].map(
            ORIGIN
        )
    )

    merged[
        "category"
    ] = (
        merged[
            "method"
        ].map(
            CATEGORY
        )
    )

    merged = merged.sort_values(
        [
            "alpha",
            "test_rmse_mean",
        ],
        ascending=[
            False,
            True,
        ],
    )

    output_csv = (
        final_dir
        / "final_with_published_baselines.csv"
    )

    merged.to_csv(
        output_csv,
        index=False,
    )

    table_dir = (
        final_dir
        / "paper_tables"
    )

    table_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Controlled comparison with standard and published federated RUL baselines on C-MAPSS FD004. Results are mean $\pm$ standard deviation over three seeds.}",
        r"\label{tab:published_baselines}",
        r"\begin{tabular}{llrrrr}",
        r"\hline",
        r"$\alpha$ & Method & Val. RMSE & Test RMSE & Test MAE & Test $R^2$ \\",
        r"\hline",
    ]

    for alpha in sorted(
        merged[
            "alpha"
        ].unique(),
        reverse=True,
    ):
        block = merged[
            merged[
                "alpha"
            ] == alpha
        ]

        for _, row in (
            block.iterrows()
        ):
            lines.append(
                f"{alpha} & "
                f"{row['display_name']} & "
                f"{fmt(row['val_rmse_mean'], row['val_rmse_std'])} & "
                f"{fmt(row['test_rmse_mean'], row['test_rmse_std'])} & "
                f"{fmt(row['test_mae_mean'], row['test_mae_std'])} & "
                f"{fmt(row['test_r2_mean'], row['test_r2_std'])} \\\\"
            )

        lines.append(
            r"\hline"
        )

    lines.extend([
        r"\end{tabular}",
        r"\end{table*}",
    ])

    latex_path = (
        table_dir
        / "final_with_published_baselines.tex"
    )

    latex_path.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    print(
        f"Final merged CSV: "
        f"{output_csv}"
    )

    print(
        f"LaTeX table: "
        f"{latex_path}"
    )

    print(
        "\nTEST RMSE RANKING"
    )

    for alpha in sorted(
        merged[
            "alpha"
        ].unique(),
        reverse=True,
    ):
        print(
            f"\nalpha={alpha}"
        )

        block = (
            merged[
                merged[
                    "alpha"
                ] == alpha
            ]
            .sort_values(
                "test_rmse_mean"
            )
        )

        for _, row in (
            block.iterrows()
        ):
            print(
                f"  "
                f"{row['display_name']:<34} "
                f"{row['test_rmse_mean']:.3f} "
                f"± "
                f"{row['test_rmse_std']:.3f}"
            )


if __name__ == "__main__":
    main()
