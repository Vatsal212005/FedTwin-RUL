import argparse
from pathlib import Path

import pandas as pd


LABELS = {
    "centralized_transformer": "Centralized Transformer",
    "local_only_transformer": "Local-only Transformer",
    "fedavg_transformer": "FedAvg Transformer",
    "fedprox_transformer": "FedProx Transformer",
    "fedper_transformer": "FedPer-style Transformer",
    "fedavg_moe": "FedAvg-MoE",
    "twinfed_moe": "TwinFed-MoE (Proposed)",
    "ablation_no_dt": "w/o Digital Twin",
    "ablation_no_moe": "w/o MoE",
    "ablation_no_personalization": "w/o Personalization",
    "ablation_no_tgc": "w/o TGC",
    "ablation_fedavg_aggregation": "FedAvg Aggregation",
}


def resolve_suite(
    suite_dir,
    outputs_root,
):
    if suite_dir:
        return Path(suite_dir)

    pointer = (
        Path(outputs_root)
        / "latest_benchmark.txt"
    )

    return Path(
        pointer.read_text(
            encoding="utf-8"
        ).strip()
    )


def fnum(
    value,
    digits=2,
):
    if pd.isna(value):
        return "--"

    return format(
        float(value),
        f".{digits}f",
    )


def escape_latex(
    text,
):
    return (
        str(text)
        .replace(
            "_",
            r"\_",
        )
        .replace(
            "%",
            r"\%",
        )
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
        / "benchmark_results.csv"
    )

    df["label"] = (
        df["method"]
        .map(LABELS)
        .fillna(
            df["method"]
        )
    )

    paper_dir = (
        suite
        / "paper_tables"
    )

    paper_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    main_df = df[
        df["category"].isin(
            [
                "reference",
                "baseline",
                "proposed",
            ]
        )
    ].copy()

    main_df[
        [
            "label",
            "test_rmse",
            "test_mae",
            "test_r2",
            "test_nasa",
            "train_seconds",
            "parameters",
        ]
    ].to_csv(
        paper_dir
        / "main_comparison.csv",
        index=False,
    )

    ablation_df = df[
        df["category"].isin(
            [
                "proposed",
                "ablation",
            ]
        )
    ].copy()

    proposed = df[
        df["method"]
        == "twinfed_moe"
    ]

    if len(proposed):
        proposed_rmse = float(
            proposed.iloc[0][
                "test_rmse"
            ]
        )

        ablation_df[
            "delta_test_rmse"
        ] = (
            ablation_df[
                "test_rmse"
            ]
            - proposed_rmse
        )
    else:
        ablation_df[
            "delta_test_rmse"
        ] = float("nan")

    ablation_df[
        [
            "label",
            "val_rmse",
            "test_rmse",
            "test_mae",
            "test_r2",
            "delta_test_rmse",
        ]
    ].to_csv(
        paper_dir
        / "ablation_comparison.csv",
        index=False,
    )

    main_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Comparison on the C-MAPSS FD004 dataset. Lower RMSE, MAE, and NASA score are better; higher $R^2$ is better.}",
        r"\label{tab:fd004_main}",
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        r"Method & RMSE & MAE & $R^2$ & NASA Score & Time (s) \\",
        r"\hline",
    ]

    for _, row in main_df.iterrows():
        main_lines.append(
            f"{escape_latex(row['label'])} & "
            f"{fnum(row['test_rmse'])} & "
            f"{fnum(row['test_mae'])} & "
            f"{fnum(row['test_r2'], 3)} & "
            f"{fnum(row['test_nasa'], 1)} & "
            f"{fnum(row['train_seconds'], 1)} \\\\"
        )

    main_lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    (
        paper_dir
        / "main_comparison.tex"
    ).write_text(
        "\n".join(
            main_lines
        ),
        encoding="utf-8",
    )

    ablation_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study of TwinFed-MoE on FD004.}",
        r"\label{tab:fd004_ablation}",
        r"\begin{tabular}{lrr}",
        r"\hline",
        r"Variant & Test RMSE & $\Delta$RMSE \\",
        r"\hline",
    ]

    for _, row in (
        ablation_df.iterrows()
    ):
        ablation_lines.append(
            f"{escape_latex(row['label'])} & "
            f"{fnum(row['test_rmse'])} & "
            f"{fnum(row['delta_test_rmse'])} \\\\"
        )

    ablation_lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])

    (
        paper_dir
        / "ablation_comparison.tex"
    ).write_text(
        "\n".join(
            ablation_lines
        ),
        encoding="utf-8",
    )

    print(
        f"Saved paper tables to "
        f"{paper_dir}"
    )


if __name__ == "__main__":
    main()
