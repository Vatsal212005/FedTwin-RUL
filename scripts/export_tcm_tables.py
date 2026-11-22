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
    "tcm_identity_modulation": "TCM Identity Control",
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


def fmt(
    value,
    digits=2,
):
    if pd.isna(
        value
    ):
        return "--"

    return format(
        float(
            value
        ),
        f".{digits}f",
    )


def latex_escape(
    text,
):
    return (
        str(
            text
        )
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

    merged_path = (
        suite
        / "tcm_vs_fixed_baselines.csv"
    )

    if merged_path.exists():
        df = pd.read_csv(
            merged_path
        )
    else:
        df = pd.read_csv(
            suite
            / "tcm_results.csv"
        )

    df[
        "label"
    ] = (
        df[
            "method"
        ]
        .map(
            LABELS
        )
        .fillna(
            df[
                "method"
            ]
        )
    )

    out_dir = (
        suite
        / "paper_tables"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cols = [
        "label",
        "val_rmse",
        "test_rmse",
        "test_mae",
        "test_r2",
        "test_nasa",
        "train_seconds",
        "parameters",
    ]

    df[
        [
            c
            for c in cols
            if c in df.columns
        ]
    ].to_csv(
        out_dir
        / "tcm_main_comparison.csv",
        index=False,
    )

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Comparison of TwinFed-TCM and reference methods on C-MAPSS FD004.}",
        r"\label{tab:tcm_fd004}",
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        r"Method & Val. RMSE & Test RMSE & Test MAE & $R^2$ & NASA \\",
        r"\hline",
    ]

    for _, row in (
        df.iterrows()
    ):
        lines.append(
            f"{latex_escape(row['label'])} & "
            f"{fmt(row.get('val_rmse'))} & "
            f"{fmt(row.get('test_rmse'))} & "
            f"{fmt(row.get('test_mae'))} & "
            f"{fmt(row.get('test_r2'), 3)} & "
            f"{fmt(row.get('test_nasa'), 1)} \\\\"
        )

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    (
        out_dir
        / "tcm_main_comparison.tex"
    ).write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    print(
        f"Saved TCM paper tables to "
        f"{out_dir}"
    )


if __name__ == "__main__":
    main()
