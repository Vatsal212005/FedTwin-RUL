import argparse
import json
from pathlib import Path

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

    best_rmse = df.loc[
        df["pooled_rmse"].idxmin()
    ]

    best_nasa = df.loc[
        df["pooled_nasa_score"].idxmin()
    ]

    best_worst_client = df.loc[
        df["worst_client_rmse"].idxmin()
    ]

    best_r2 = df.loc[
        df["pooled_r2"].idxmax()
    ]

    print(
        f"Run: {run_dir}"
    )

    print(
        "\nBest pooled RMSE:"
    )

    print(
        f"  round="
        f"{int(best_rmse['round'])}, "
        f"RMSE="
        f"{best_rmse['pooled_rmse']:.4f}, "
        f"MAE="
        f"{best_rmse['pooled_mae']:.4f}, "
        f"R2="
        f"{best_rmse['pooled_r2']:.4f}"
    )

    print(
        "\nBest pooled R2:"
    )

    print(
        f"  round="
        f"{int(best_r2['round'])}, "
        f"R2="
        f"{best_r2['pooled_r2']:.4f}, "
        f"RMSE="
        f"{best_r2['pooled_rmse']:.4f}"
    )

    print(
        "\nBest NASA score:"
    )

    print(
        f"  round="
        f"{int(best_nasa['round'])}, "
        f"score="
        f"{best_nasa['pooled_nasa_score']:.4f}"
    )

    print(
        "\nBest worst-client RMSE:"
    )

    print(
        f"  round="
        f"{int(best_worst_client['round'])}, "
        f"worst-client RMSE="
        f"{best_worst_client['worst_client_rmse']:.4f}"
    )

    print(
        "\nPersonalization at best-RMSE round:"
    )

    print(
        f"  personalized pooled RMSE="
        f"{best_rmse['pooled_rmse']:.4f}"
    )

    print(
        f"  shared-only pooled RMSE="
        f"{best_rmse['shared_pooled_rmse']:.4f}"
    )

    print(
        f"  RMSE gain="
        f"{best_rmse['personalization_gain_rmse']:.4f}"
    )

    official_path = (
        run_dir
        / "official_test_metrics.json"
    )

    if official_path.exists():
        official = json.loads(
            official_path.read_text(
                encoding="utf-8"
            )
        )

        m = official[
            "metrics"
        ]

        print(
            "\nOfficial FD004 test:"
        )

        print(
            f"  RMSE="
            f"{m['rmse']:.4f}"
        )

        print(
            f"  MAE="
            f"{m['mae']:.4f}"
        )

        print(
            f"  R2="
            f"{m['r2']:.4f}"
        )

        print(
            f"  NASA="
            f"{m['nasa_score']:.4f}"
        )


if __name__ == "__main__":
    main()
