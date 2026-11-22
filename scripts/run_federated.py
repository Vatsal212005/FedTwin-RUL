import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twinfed_moe.utils.config import load_config
from twinfed_moe.utils.seed import set_seed
from twinfed_moe.utils.device import resolve_device
from twinfed_moe.data.synthetic import make_synthetic_rul_dataset
from twinfed_moe.data.cmapss import load_processed_cmapss
from twinfed_moe.data.partition import (
    dirichlet_partition,
    balanced_regime_partition,
    split_subset_by_unit,
)
from twinfed_moe.models.twinfed_moe import TwinFedMoE
from twinfed_moe.federated.client import FederatedClient
from twinfed_moe.federated.server import FederatedServer


def fmt_seconds(seconds):
    seconds = int(round(seconds))

    if seconds < 60:
        return f"{seconds}s"

    minutes, sec = divmod(
        seconds,
        60,
    )

    if minutes < 60:
        return (
            f"{minutes}m "
            f"{sec:02d}s"
        )

    hours, minutes = divmod(
        minutes,
        60,
    )

    return (
        f"{hours}h "
        f"{minutes:02d}m "
        f"{sec:02d}s"
    )


def better(value, best, mode):
    if best is None:
        return True

    if mode == "min":
        return value < best

    if mode == "max":
        return value > best

    raise ValueError(
        "checkpoint_mode must be 'min' or 'max'."
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/fd004.yaml",
    )

    args = parser.parse_args()

    cfg = load_config(
        args.config
    )

    set_seed(
        cfg["seed"]
    )

    device = resolve_device(
        cfg.get(
            "device",
            "auto",
        )
    )

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":
        print(
            "GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    dataset_name = cfg[
        "data"
    ]["dataset"]

    if dataset_name == "synthetic":
        dataset = make_synthetic_rul_dataset(
            num_samples=cfg["data"]["num_samples"],
            seq_len=cfg["data"]["seq_len"],
            num_sensors=cfg["data"]["num_sensors"],
            seed=cfg["seed"],
        )

        client_sets = dirichlet_partition(
            dataset,
            num_clients=cfg["data"]["num_clients"],
            alpha=cfg["data"]["non_iid_alpha"],
            seed=cfg["seed"],
        )

        train_val = [
            (s, s)
            for s in client_sets
        ]

        twin_aux_dim = (
            dataset[0][1].numel()
        )

    elif dataset_name == "cmapss":
        dataset = load_processed_cmapss(
            cfg["data"]["train_path"]
        )

        actual_sensors = int(
            dataset.X.shape[-1]
        )

        expected_sensors = int(
            cfg["data"]["num_sensors"]
        )

        if actual_sensors != expected_sensors:
            raise ValueError(
                f"Config expects "
                f"{expected_sensors} sensors, "
                f"but processed data contains "
                f"{actual_sensors}."
            )

        client_sets = balanced_regime_partition(
            dataset,
            num_clients=cfg["data"]["num_clients"],
            alpha=cfg["data"]["non_iid_alpha"],
            seed=cfg["seed"],
            affinity_weight=cfg["data"].get(
                "regime_affinity_weight",
                0.20,
            ),
        )

        train_val = [
            split_subset_by_unit(
                subset,
                val_fraction=cfg["data"].get(
                    "val_fraction",
                    0.2,
                ),
                seed=cfg["seed"] + i,
            )
            for i, subset
            in enumerate(client_sets)
        ]

        twin_aux_dim = int(
            dataset.twin_aux.shape[-1]
        )

        print(
            f"Loaded {cfg['data']['subset']}: "
            f"{len(dataset):,} windows, "
            f"{len(set(dataset.unit_ids.tolist()))} engines"
        )

        train_sizes = [
            len(x[0])
            for x in train_val
        ]

        val_sizes = [
            len(x[1])
            for x in train_val
        ]

        totals = [
            a + b
            for a, b
            in zip(
                train_sizes,
                val_sizes,
            )
        ]

        print(
            "Client train windows:",
            train_sizes,
        )

        print(
            "Client validation windows:",
            val_sizes,
        )

        print(
            "Client total-window imbalance ratio: "
            f"{max(totals) / max(1, min(totals)):.2f}x"
        )

    else:
        raise ValueError(
            f"Unknown dataset: "
            f"{dataset_name}"
        )

    def model_factory():
        m = cfg["model"]

        return TwinFedMoE(
            num_sensors=cfg["data"]["num_sensors"],
            twin_aux_dim=twin_aux_dim,
            seq_len=cfg["data"]["seq_len"],
            d_model=m["d_model"],
            nhead=m["nhead"],
            transformer_layers=m["transformer_layers"],
            dim_feedforward=m["dim_feedforward"],
            num_experts=m["num_experts"],
            top_k=m["top_k"],
            twin_latent_dim=m["twin_latent_dim"],
            expert_hidden_dim=m["expert_hidden_dim"],
            adapter_hidden_dim=m["adapter_hidden_dim"],
            dropout=m["dropout"],
        )

    global_model = (
        model_factory().cpu()
    )

    clients = [
        FederatedClient(
            client_id=i,
            train_dataset=train_ds,
            val_dataset=val_ds,
            model_factory=model_factory,
            cfg=cfg,
            device=device,
        )
        for i, (
            train_ds,
            val_ds,
        ) in enumerate(train_val)
    ]

    server = FederatedServer(
        global_model,
        clients,
        cfg,
    )

    output_root = Path(
        cfg.get(
            "output_dir",
            "outputs",
        )
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    subset_name = cfg[
        "data"
    ].get(
        "subset",
        "smoke",
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    run_name = (
        f"TwinFedMoE_"
        f"{subset_name}_"
        f"{timestamp}"
    )

    run_dir = (
        output_root
        / run_name
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    # Helpful pointer for evaluation scripts.
    (
        output_root
        / "latest_run.txt"
    ).write_text(
        str(run_dir.resolve()),
        encoding="utf-8",
    )

    (
        run_dir
        / "config_snapshot.yaml"
    ).write_text(
        yaml.safe_dump(
            cfg,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    metrics_path = (
        run_dir
        / "round_metrics.csv"
    )

    best_path = (
        run_dir
        / "best_model.pt"
    )

    last_path = (
        run_dir
        / "last_model.pt"
    )

    summary_path = (
        run_dir
        / "run_summary.json"
    )

    num_experts = int(
        cfg["model"]["num_experts"]
    )

    fieldnames = [
        "round",
        "mean_train_loss",

        "pooled_rmse",
        "pooled_mae",
        "pooled_r2",
        "pooled_nasa_score",

        "mean_client_rmse",
        "mean_client_mae",
        "mean_client_r2",
        "worst_client_rmse",

        "shared_pooled_rmse",
        "shared_pooled_mae",
        "shared_pooled_r2",
        "shared_pooled_nasa_score",

        "personalization_gain_rmse",

        "expert_utilization_entropy",

        *[
            f"expert_util_{i}"
            for i in range(
                num_experts
            )
        ],

        "round_seconds",
    ]

    total_rounds = int(
        cfg["training"]["rounds"]
    )

    checkpoint_metric = cfg.get(
        "evaluation",
        {},
    ).get(
        "checkpoint_metric",
        "pooled_rmse",
    )

    checkpoint_mode = cfg.get(
        "evaluation",
        {},
    ).get(
        "checkpoint_mode",
        "min",
    )

    best_value = None
    best_round = None
    best_metrics = None

    run_start = time.perf_counter()
    round_times = []

    print(
        f"Run directory: "
        f"{run_dir}"
    )

    print(
        f"Checkpoint criterion: "
        f"{checkpoint_metric} "
        f"({checkpoint_mode})"
    )

    with metrics_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for r in range(
            1,
            total_rounds + 1,
        ):
            metrics = server.run_round(
                r
            )

            writer.writerow(
                metrics
            )

            f.flush()

            checkpoint = (
                server.checkpoint_payload(
                    r,
                    metrics,
                )
            )

            # Crash-safe last checkpoint updated each round.
            torch.save(
                checkpoint,
                last_path,
            )

            current_value = float(
                metrics[
                    checkpoint_metric
                ]
            )

            if better(
                current_value,
                best_value,
                checkpoint_mode,
            ):
                best_value = (
                    current_value
                )

                best_round = (
                    r
                )

                best_metrics = dict(
                    metrics
                )

                torch.save(
                    checkpoint,
                    best_path,
                )

                best_flag = " | NEW BEST"
            else:
                best_flag = ""

            round_times.append(
                metrics[
                    "round_seconds"
                ]
            )

            mean_round = (
                sum(round_times)
                / len(round_times)
            )

            eta = (
                total_rounds - r
            ) * mean_round

            elapsed = (
                time.perf_counter()
                - run_start
            )

            print(
                f"Round {r:03d} complete | "
                f"pooled RMSE="
                f"{metrics['pooled_rmse']:.3f} | "
                f"MAE="
                f"{metrics['pooled_mae']:.3f} | "
                f"R2="
                f"{metrics['pooled_r2']:.3f} | "
                f"shared RMSE="
                f"{metrics['shared_pooled_rmse']:.3f} | "
                f"personalization gain="
                f"{metrics['personalization_gain_rmse']:.3f} | "
                f"worst client="
                f"{metrics['worst_client_rmse']:.3f} | "
                f"round="
                f"{fmt_seconds(metrics['round_seconds'])} | "
                f"elapsed="
                f"{fmt_seconds(elapsed)} | "
                f"ETA="
                f"{fmt_seconds(eta)}"
                f"{best_flag}",
                flush=True,
            )

    total_elapsed = (
        time.perf_counter()
        - run_start
    )

    run_summary = {
        "run_name": run_name,
        "dataset": dataset_name,
        "subset": subset_name,
        "seed": int(
            cfg["seed"]
        ),
        "rounds": total_rounds,
        "checkpoint_metric": checkpoint_metric,
        "checkpoint_mode": checkpoint_mode,
        "best_round": int(
            best_round
        ),
        "best_value": float(
            best_value
        ),
        "best_metrics": best_metrics,
        "total_seconds": float(
            total_elapsed
        ),
        "metrics_csv": str(
            metrics_path
        ),
        "best_checkpoint": str(
            best_path
        ),
        "last_checkpoint": str(
            last_path
        ),
        "official_test_evaluated": False,
    }

    summary_path.write_text(
        json.dumps(
            run_summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\nTraining finished in "
        f"{fmt_seconds(total_elapsed)}"
    )

    print(
        f"Best round: "
        f"{best_round} | "
        f"{checkpoint_metric}="
        f"{best_value:.4f}"
    )

    print(
        f"Best checkpoint: "
        f"{best_path}"
    )

    print(
        f"Last checkpoint: "
        f"{last_path}"
    )

    print(
        f"Metrics: "
        f"{metrics_path}"
    )

    print(
        f"Summary: "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()
