import argparse
import copy
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twinfed_moe.data.cmapss import load_processed_cmapss
from twinfed_moe.data.partition import balanced_regime_partition, split_subset_by_unit
from twinfed_moe.federated.tcm_aggregation import fedavg_state_dicts
from twinfed_moe.models.benchmark_model import BenchmarkRULModel
from twinfed_moe.models.tcm_model import TwinFedTCM
from twinfed_moe.training.metrics import metric_dict
from twinfed_moe.training.tcm_trainer import predict_tcm, train_local_tcm
from twinfed_moe.training.trainer import predict as predict_benchmark
from twinfed_moe.training.trainer import train_local as train_benchmark
from twinfed_moe.utils.config import load_config
from twinfed_moe.utils.device import resolve_device
from twinfed_moe.utils.seed import set_seed


METHODS = {
    "fedavg_transformer": {
        "family": "transformer",
        "prox_mu": 0.0,
        "description": "FedAvg temporal Transformer baseline.",
    },
    "fedprox_transformer": {
        "family": "transformer",
        "prox_mu": 0.001,
        "description": "FedProx temporal Transformer baseline.",
    },
    "fedavg_moe": {
        "family": "moe",
        "prox_mu": 0.0,
        "description": "Sparse Top-2 temporal MoE with FedAvg.",
    },
    "fedtwin_rul": {
        "family": "fedtwin",
        "prox_mu": 0.0,
        "description": (
            "FedTwin-RUL: lightweight Digital-Twin state concatenated "
            "with the temporal Transformer representation and trained with FedAvg."
        ),
    },
}


def clone_cpu_state(model):
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


def make_loader(dataset, cfg, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
    )


def build_model(cfg, method_name, twin_aux_dim):
    spec = METHODS[method_name]
    m = cfg["model"]

    if spec["family"] == "fedtwin":
        return TwinFedTCM(
            num_sensors=cfg["data"]["num_sensors"],
            twin_aux_dim=twin_aux_dim,
            seq_len=cfg["data"]["seq_len"],
            d_model=m["d_model"],
            nhead=m["nhead"],
            transformer_layers=m["transformer_layers"],
            dim_feedforward=m["dim_feedforward"],
            twin_latent_dim=m["twin_latent_dim"],
            fusion_hidden_dim=m["fusion_hidden_dim"],
            dropout=m["dropout"],
            fusion_mode="concat",
            use_dt_reconstruction=True,
        )

    use_moe = spec["family"] == "moe"

    return BenchmarkRULModel(
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
        use_digital_twin=False,
        use_moe=use_moe,
        use_personalization=False,
        router_mode="temporal",
    )


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters()))


def method_cfg(base_cfg, method_name):
    cfg = copy.deepcopy(base_cfg)
    spec = METHODS[method_name]

    cfg["training"]["prox_mu"] = float(spec["prox_mu"])

    if spec["family"] == "fedtwin":
        cfg["training"]["lambda_dt"] = float(
            base_cfg["training"].get("lambda_dt", 0.10)
        )
        cfg["training"]["lambda_bal"] = 0.0
        cfg["training"]["lambda_tgc"] = 0.0
    elif spec["family"] == "moe":
        cfg["training"]["lambda_dt"] = 0.0
        cfg["training"]["lambda_bal"] = float(
            base_cfg["training"].get("lambda_bal", 0.01)
        )
        cfg["training"]["lambda_tgc"] = 0.0
    else:
        cfg["training"]["lambda_dt"] = 0.0
        cfg["training"]["lambda_bal"] = 0.0
        cfg["training"]["lambda_tgc"] = 0.0

    return cfg



def train_dispatch(
    model,
    loader,
    optimizer,
    device,
    cfg,
    global_state,
    method_name,
):
    if METHODS[method_name]["family"] == "fedtwin":
        return train_local_tcm(
            model,
            loader,
            optimizer,
            device,
            cfg,
            global_state=global_state,
        )

    return train_benchmark(
        model,
        loader,
        optimizer,
        device,
        cfg,
        global_state=global_state,
    )


def predict_dispatch(
    model,
    loader,
    device,
    cfg,
    method_name,
):
    if METHODS[method_name]["family"] == "fedtwin":
        result = predict_tcm(
            model,
            loader,
            device,
            cfg,
            collect_modulation=False,
        )
        return {
            "pred": result["pred"],
            "target": result["target"],
        }

    result = predict_benchmark(
        model,
        loader,
        device,
        cfg,
        use_personal_adapter=False,
        collect_routing=False,
    )

    return {
        "pred": result["pred"],
        "target": result["target"],
    }


def evaluate_pooled(models, loaders, device, cfg, method_name):
    preds = []
    targets = []
    client_metrics = []

    for model, loader in zip(models, loaders):
        result = predict_dispatch(
            model,
            loader,
            device,
            cfg,
            method_name,
        )

        metrics = metric_dict(
            result["pred"],
            result["target"],
        )

        preds.append(result["pred"])
        targets.append(result["target"])
        client_metrics.append(metrics)

    pooled = metric_dict(
        torch.cat(preds),
        torch.cat(targets),
    )

    return {
        "pooled_rmse": pooled["rmse"],
        "pooled_mae": pooled["mae"],
        "pooled_r2": pooled["r2"],
        "pooled_nasa": pooled["nasa_score"],
        "mean_client_rmse": float(
            np.mean([m["rmse"] for m in client_metrics])
        ),
        "worst_client_rmse": float(
            np.max([m["rmse"] for m in client_metrics])
        ),
    }


def run_one(
    method_name,
    alpha,
    seed,
    base_cfg,
    train_dataset,
    test_dataset,
    twin_aux_dim,
    device,
    run_dir,
):
    cfg = method_cfg(base_cfg, method_name)

    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    client_sets = balanced_regime_partition(
        train_dataset,
        num_clients=cfg["data"]["num_clients"],
        alpha=float(alpha),
        seed=int(seed),
        affinity_weight=cfg["data"].get("regime_affinity_weight", 0.20),
    )

    splits = [
        split_subset_by_unit(
            subset,
            val_fraction=cfg["data"].get("val_fraction", 0.2),
            seed=int(seed) + i,
        )
        for i, subset in enumerate(client_sets)
    ]

    train_sets = [x[0] for x in splits]
    val_sets = [x[1] for x in splits]

    train_loaders = [
        make_loader(ds, cfg, shuffle=True)
        for ds in train_sets
    ]

    val_loaders = [
        make_loader(ds, cfg, shuffle=False)
        for ds in val_sets
    ]

    test_loader = make_loader(
        test_dataset,
        cfg,
        shuffle=False,
    )

    global_model = build_model(
        cfg,
        method_name,
        twin_aux_dim,
    ).cpu()

    client_models = [
        build_model(
            cfg,
            method_name,
            twin_aux_dim,
        ).to(device)
        for _ in train_sets
    ]

    best_rmse = None
    best_round = None
    best_state = None
    best_validation = None

    round_rows = []

    start = time.perf_counter()

    for round_idx in range(
        1,
        int(cfg["training"]["rounds"]) + 1,
    ):
        global_state = clone_cpu_state(
            global_model
        )

        states = []
        sample_counts = []
        losses = []

        for cid, model in enumerate(client_models):
            model.load_state_dict(
                global_state,
                strict=True,
            )

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg["training"]["lr"],
                weight_decay=cfg["training"]["weight_decay"],
            )

            result = train_dispatch(
                model,
                train_loaders[cid],
                optimizer,
                device,
                cfg,
                global_state,
                method_name,
            )

            states.append(
                clone_cpu_state(model)
            )

            sample_counts.append(
                len(train_sets[cid])
            )

            losses.append(
                result["loss"]
            )

        aggregated = fedavg_state_dicts(
            states,
            sample_counts,
        )

        global_model.load_state_dict(
            aggregated,
            strict=True,
        )

        for model in client_models:
            model.load_state_dict(
                aggregated,
                strict=True,
            )

        validation = evaluate_pooled(
            client_models,
            val_loaders,
            device,
            cfg,
            method_name,
        )

        round_rows.append({
            "round": round_idx,
            "mean_train_loss": float(
                np.mean(losses)
            ),
            **validation,
        })

        current = float(
            validation["pooled_rmse"]
        )

        if (
            best_rmse is None
            or current < best_rmse
        ):
            best_rmse = current
            best_round = int(round_idx)
            best_state = clone_cpu_state(
                global_model
            )
            best_validation = dict(
                validation
            )

        if (
            round_idx == 1
            or round_idx % 10 == 0
            or round_idx
            == int(cfg["training"]["rounds"])
        ):
            print(
                f"    round {round_idx:02d}/"
                f"{cfg['training']['rounds']} | "
                f"val RMSE={current:.3f} | "
                f"best={best_rmse:.3f}",
                flush=True,
            )

    train_seconds = (
        time.perf_counter()
        - start
    )

    test_model = build_model(
        cfg,
        method_name,
        twin_aux_dim,
    )

    test_model.load_state_dict(
        best_state,
        strict=True,
    )

    test_model = test_model.to(
        device
    )

    test_result = predict_dispatch(
        test_model,
        test_loader,
        device,
        cfg,
        method_name,
    )

    test_metrics = metric_dict(
        test_result["pred"],
        test_result["target"],
    )

    pd.DataFrame(
        round_rows
    ).to_csv(
        run_dir / "round_metrics.csv",
        index=False,
    )

    torch.save(
        {
            "method": method_name,
            "alpha": float(alpha),
            "seed": int(seed),
            "best_round": best_round,
            "validation": best_validation,
            "global_model_state": best_state,
            "config": cfg,
        },
        run_dir / "best_model.pt",
    )

    result = {
        "method": method_name,
        "description": METHODS[
            method_name
        ]["description"],
        "alpha": float(alpha),
        "seed": int(seed),
        "best_round": best_round,

        "val_rmse": best_validation[
            "pooled_rmse"
        ],
        "val_mae": best_validation[
            "pooled_mae"
        ],
        "val_r2": best_validation[
            "pooled_r2"
        ],
        "val_nasa": best_validation[
            "pooled_nasa"
        ],
        "mean_client_rmse": best_validation[
            "mean_client_rmse"
        ],
        "worst_client_rmse": best_validation[
            "worst_client_rmse"
        ],

        "test_rmse": test_metrics[
            "rmse"
        ],
        "test_mae": test_metrics[
            "mae"
        ],
        "test_r2": test_metrics[
            "r2"
        ],
        "test_nasa": test_metrics[
            "nasa_score"
        ],

        "train_seconds": float(
            train_seconds
        ),
        "parameters": count_parameters(
            test_model
        ),
    }

    (
        run_dir / "result.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    return result


def build_tables(study_dir):
    rows = []

    for path in study_dir.glob(
        "runs/*/*/*/result.json"
    ):
        rows.append(
            json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        )

    raw = pd.DataFrame(rows)

    raw_path = (
        study_dir
        / "raw_results.csv"
    )

    raw.to_csv(
        raw_path,
        index=False,
    )

    if raw.empty:
        return raw_path, None

    grouped = []

    for (
        method,
        alpha,
    ), group in raw.groupby(
        [
            "method",
            "alpha",
        ],
        sort=True,
    ):
        row = {
            "method": method,
            "alpha": float(alpha),
            "n_seeds": int(
                len(group)
            ),
        }

        for metric in [
            "val_rmse",
            "val_mae",
            "val_r2",
            "mean_client_rmse",
            "worst_client_rmse",
            "test_rmse",
            "test_mae",
            "test_r2",
            "test_nasa",
            "train_seconds",
            "parameters",
        ]:
            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            )

            row[
                f"{metric}_mean"
            ] = float(
                values.mean()
            )

            row[
                f"{metric}_std"
            ] = float(
                values.std(
                    ddof=1
                )
                if len(values) > 1
                else 0.0
            )

        grouped.append(row)

    summary = pd.DataFrame(
        grouped
    )

    summary_path = (
        study_dir
        / "mean_std_results.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    return (
        raw_path,
        summary_path,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/fedtwin_final.yaml",
    )

    parser.add_argument(
        "--study-dir",
        default=None,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
    )

    parser.add_argument(
        "--alphas",
        nargs="*",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    cfg = load_config(
        args.config
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

    train_dataset = load_processed_cmapss(
        cfg["data"]["train_path"]
    )

    test_dataset = load_processed_cmapss(
        cfg["data"]["test_path"]
    )

    twin_aux_dim = int(
        train_dataset.twin_aux.shape[-1]
    )

    methods = (
        args.methods
        if args.methods
        else cfg["study"]["methods"]
    )

    alphas = (
        args.alphas
        if args.alphas
        else cfg["study"]["alphas"]
    )

    seeds = (
        args.seeds
        if args.seeds
        else cfg["study"]["seeds"]
    )

    unknown = [
        m
        for m in methods
        if m not in METHODS
    ]

    if unknown:
        raise ValueError(
            f"Unknown methods: {unknown}"
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

    if args.study_dir:
        study_dir = Path(
            args.study_dir
        )
    else:
        stamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        study_dir = (
            output_root
            / f"fedtwin_final_FD004_{stamp}"
        )

    study_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        output_root
        / "latest_fedtwin_final.txt"
    ).write_text(
        str(study_dir.resolve()),
        encoding="utf-8",
    )

    total_jobs = (
        len(methods)
        * len(alphas)
        * len(seeds)
    )

    job_idx = 0
    start = time.perf_counter()

    for alpha in alphas:
        for seed in seeds:
            for method in methods:
                job_idx += 1

                run_dir = (
                    study_dir
                    / "runs"
                    / f"alpha_{alpha}"
                    / f"seed_{seed}"
                    / method
                )

                run_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                result_path = (
                    run_dir
                    / "result.json"
                )

                if (
                    args.resume
                    and result_path.exists()
                ):
                    print(
                        f"[{job_idx}/{total_jobs}] "
                        f"skip {method}, "
                        f"alpha={alpha}, seed={seed}"
                    )
                    continue

                print(
                    f"\n[{job_idx}/{total_jobs}] "
                    f"{method} | "
                    f"alpha={alpha} | seed={seed}"
                )

                result = run_one(
                    method,
                    alpha,
                    seed,
                    cfg,
                    train_dataset,
                    test_dataset,
                    twin_aux_dim,
                    device,
                    run_dir,
                )

                print(
                    f"  done | "
                    f"val RMSE={result['val_rmse']:.3f} | "
                    f"test RMSE={result['test_rmse']:.3f} | "
                    f"time={result['train_seconds']:.1f}s"
                )

                build_tables(
                    study_dir
                )

    elapsed = (
        time.perf_counter()
        - start
    )

    raw_path, summary_path = (
        build_tables(
            study_dir
        )
    )

    summary = {
        "study_dir": str(
            study_dir
        ),
        "methods": methods,
        "alphas": alphas,
        "seeds": seeds,
        "total_jobs": total_jobs,
        "elapsed_seconds_this_invocation": float(
            elapsed
        ),
        "raw_results_csv": str(
            raw_path
        ),
        "mean_std_results_csv": (
            str(summary_path)
            if summary_path
            else None
        ),
    }

    (
        study_dir
        / "study_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nFinal FedTwin-RUL study complete."
    )
    print(
        f"Raw results: {raw_path}"
    )
    print(
        f"Mean/std results: {summary_path}"
    )
    print(
        f"Elapsed: {elapsed / 60.0:.2f} min"
    )


if __name__ == "__main__":
    main()
