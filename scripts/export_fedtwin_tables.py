import argparse
from pathlib import Path
import pandas as pd


LABELS = {
    "fedavg_transformer": "FedAvg Transformer",
    "fedprox_transformer": "FedProx Transformer",
    "fedavg_moe": "FedAvg-MoE",
    "fedtwin_rul": "FedTwin-RUL",
}


def resolve_study(study_dir, outputs_root):
    if study_dir:
        return Path(study_dir)

    pointer = Path(outputs_root) / "latest_fedtwin_final.txt"
    return Path(
        pointer.read_text(
            encoding="utf-8"
        ).strip()
    )


def fmt(mean, std, digits=2):
    return (
        f"{float(mean):.{digits}f} "
        f"$\\pm$ {float(std):.{digits}f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", default=None)
    parser.add_argument("--outputs-root", default="outputs")
    args = parser.parse_args()

    study = resolve_study(
        args.study_dir,
        args.outputs_root,
    )

    df = pd.read_csv(
        study / "mean_std_results.csv"
    )

    out_dir = study / "paper_tables"
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        out_dir / "fedtwin_mean_std.csv",
        index=False,
    )

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Performance under different client heterogeneity levels on C-MAPSS FD004. Results are mean $\pm$ standard deviation over three seeds.}",
        r"\label{tab:fedtwin_heterogeneity}",
        r"\begin{tabular}{llrrrr}",
        r"\hline",
        r"$\alpha$ & Method & Val. RMSE & Test RMSE & Test MAE & Test $R^2$ \\",
        r"\hline",
    ]

    for alpha in sorted(
        df["alpha"].unique(),
        reverse=True,
    ):
        block = df[
            df["alpha"] == alpha
        ]

        for _, row in block.iterrows():
            label = LABELS.get(
                row["method"],
                row["method"],
            )

            lines.append(
                f"{alpha} & {label} & "
                f"{fmt(row['val_rmse_mean'], row['val_rmse_std'])} & "
                f"{fmt(row['test_rmse_mean'], row['test_rmse_std'])} & "
                f"{fmt(row['test_mae_mean'], row['test_mae_std'])} & "
                f"{fmt(row['test_r2_mean'], row['test_r2_std'], 3)} \\\\"
            )

        lines.append(r"\hline")

    lines.extend([
        r"\end{tabular}",
        r"\end{table*}",
    ])

    (
        out_dir / "fedtwin_heterogeneity.tex"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(
        f"Saved paper tables to {out_dir}"
    )


if __name__ == "__main__":
    main()
