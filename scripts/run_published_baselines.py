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
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twinfed_moe.data.cmapss import load_processed_cmapss
from twinfed_moe.data.partition import (
    balanced_regime_partition,
    split_subset_by_unit,
)
from twinfed_moe.models.published_baselines import (
    StandardTransformerRUL,
    TransLighterReproduction,
)
from twinfed_moe.training.metrics import metric_dict
from twinfed_moe.utils.config import load_config
from twinfed_moe.utils.device import resolve_device
from twinfed_moe.utils.seed import set_seed


METHODS = {
    "trans_lighter_repro": {
        "display_name": "Trans-Lighter (protocol-aligned)",
        "origin": (
            "Nguyen et al., Computers in Industry 148 (2023), 103888; "
            "DOI 10.1016/j.compind.2023.103888"
        ),
        "type": "rul_specific_published",
        "algorithm": "fedavg",
        "description": (
            "Protocol-aligned reproduction of the published Trans-Lighter "
            "encoder-only federated Transformer architecture. Bayesian "
            "hyperparameter optimization from the original paper is not "
            "repeated; common study hyperparameters are used for fairness."
        ),
    },
    "scaffold_transformer": {
        "display_name": "SCAFFOLD Transformer",
        "origin": (
            "Karimireddy et al., ICML/PMLR 119 (2020), "
            "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"
        ),
        "type": "published_fl_baseline",
        "algorithm": "scaffold",
        "description": (
            "Canonical SCAFFOLD client-drift correction with a temporal "
            "Transformer RUL backbone and no Digital-Twin inputs."
        ),
    },
}


def clone_cpu_state(model):
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


def count_parameters(model):
    return int(
        sum(
            p.numel()
            for p in model.parameters()
        )
    )


def make_loader(
    dataset,
    cfg,
    shuffle=False,
):
    return DataLoader(
        dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
    )


def build_model(
    cfg,
    method_name,
):
    m = cfg["model"]

    kwargs = dict(
        num_sensors=cfg["data"]["num_sensors"],
        seq_len=cfg["data"]["seq_len"],
        d_model=m["d_model"],
        nhead=m["nhead"],
        num_layers=m["transformer_layers"],
        dim_feedforward=m["dim_feedforward"],
        dropout=m["dropout"],
    )

    if method_name == "trans_lighter_repro":
        return TransLighterReproduction(
            **kwargs
        )

    if method_name == "scaffold_transformer":
        return StandardTransformerRUL(
            **kwargs
        )

    raise ValueError(
        f"Unknown method: {method_name}"
    )


def weighted_average_states(
    states,
    sample_counts,
):
    result = {}

    total = float(
        sum(sample_counts)
    )

    for name in states[0].keys():
        first = states[0][name]

        if not (
            first.is_floating_point()
            or first.is_complex()
        ):
            result[name] = first.clone()
            continue

        avg = torch.zeros_like(
            first
        )

        for state, n_k in zip(
            states,
            sample_counts,
        ):
            avg.add_(
                state[name],
                alpha=float(n_k) / total,
            )

        result[name] = avg

    return result


def train_fedavg_local(
    model,
    loader,
    device,
    cfg,
):
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    use_amp = (
        device.type == "cuda"
        and cfg["training"].get(
            "amp",
            True,
        )
    )

    amp_dtype = (
        torch.bfloat16
        if cfg["training"].get(
            "amp_dtype",
            "float16",
        ) == "bfloat16"
        else torch.float16
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            use_amp
            and amp_dtype == torch.float16
        ),
    )

    rul_cap = float(
        cfg["data"]["rul_cap"]
    )

    loss_sum = 0.0
    total_n = 0

    for _ in range(
        int(
            cfg["training"]["local_epochs"]
        )
    ):
        for x, twin_aux, y in loader:
            x = x.to(
                device,
                non_blocking=True,
            )

            y = y.to(
                device,
                non_blocking=True,
            )

            target = torch.clamp(
                y / rul_cap,
                0.0,
                1.0,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                pred = model(
                    x,
                    twin_aux=None,
                )

                loss = F.mse_loss(
                    pred,
                    target,
                )

            if scaler.is_enabled():
                scaler.scale(
                    loss
                ).backward()

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    5.0,
                )

                scaler.step(
                    optimizer
                )
                scaler.update()
            else:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    5.0,
                )

                optimizer.step()

            bs = int(
                x.size(0)
            )

            loss_sum += (
                float(
                    loss.detach()
                )
                * bs
            )

            total_n += bs

    return (
        loss_sum
        / max(
            total_n,
            1,
        )
    )


def init_control(
    model,
):
    return {
        name: torch.zeros_like(
            param.detach().cpu()
        )
        for name, param
        in model.named_parameters()
    }


def train_scaffold_local(
    model,
    loader,
    device,
    cfg,
    global_state,
    server_control,
    client_control,
):
    """
    Standard full-participation SCAFFOLD update.

    Local optimizer uses SGD because the closed-form control-variate update
    assumes a fixed local step size.
    """
    model.train()

    lr = float(
        cfg["training"].get(
            "scaffold_lr",
            0.01,
        )
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.0,
        weight_decay=0.0,
    )

    rul_cap = float(
        cfg["data"]["rul_cap"]
    )

    server_device = {
        name: tensor.to(
            device
        )
        for name, tensor
        in server_control.items()
    }

    client_device = {
        name: tensor.to(
            device
        )
        for name, tensor
        in client_control.items()
    }

    loss_sum = 0.0
    total_n = 0
    local_steps = 0

    for _ in range(
        int(
            cfg["training"]["local_epochs"]
        )
    ):
        for x, twin_aux, y in loader:
            x = x.to(
                device,
                non_blocking=True,
            )

            y = y.to(
                device,
                non_blocking=True,
            )

            target = torch.clamp(
                y / rul_cap,
                0.0,
                1.0,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            pred = model(
                x,
                twin_aux=None,
            )

            loss = F.mse_loss(
                pred,
                target,
            )

            loss.backward()

            for name, param in (
                model.named_parameters()
            ):
                if param.grad is None:
                    continue

                param.grad.add_(
                    server_device[name]
                    - client_device[name]
                )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

            local_steps += 1

            bs = int(
                x.size(0)
            )

            loss_sum += (
                float(
                    loss.detach()
                )
                * bs
            )

            total_n += bs

    if local_steps <= 0:
        raise RuntimeError(
            "SCAFFOLD client had zero local steps."
        )

    local_state = clone_cpu_state(
        model
    )

    new_client_control = {}
    delta_control = {}

    denominator = (
        float(local_steps)
        * lr
    )

    for name, old_ci in (
        client_control.items()
    ):
        new_ci = (
            old_ci
            - server_control[name]
            + (
                global_state[name]
                - local_state[name]
            )
            / denominator
        )

        new_client_control[
            name
        ] = new_ci

        delta_control[
            name
        ] = (
            new_ci
            - old_ci
        )

    return (
        loss_sum
        / max(
            total_n,
            1,
        ),
        local_state,
        new_client_control,
        delta_control,
    )


@torch.no_grad()
def predict(
    model,
    loader,
    device,
    cfg,
):
    model.eval()

    preds = []
    targets = []

    rul_cap = float(
        cfg["data"]["rul_cap"]
    )

    for x, twin_aux, y in loader:
        x = x.to(
            device,
            non_blocking=True,
        )

        pred = (
            model(
                x,
                twin_aux=None,
            )
            .float()
            .cpu()
            * rul_cap
        )

        preds.append(
            pred
        )

        targets.append(
            y.float().cpu()
        )

    return {
        "pred": torch.cat(
            preds
        ),
        "target": torch.cat(
            targets
        ),
    }


def evaluate_pooled(
    models,
    loaders,
    device,
    cfg,
):
    preds = []
    targets = []
    client_rmse = []

    for model, loader in zip(
        models,
        loaders,
    ):
        result = predict(
            model,
            loader,
            device,
            cfg,
        )

        metrics = metric_dict(
            result["pred"],
            result["target"],
        )

        preds.append(
            result["pred"]
        )

        targets.append(
            result["target"]
        )

        client_rmse.append(
            metrics["rmse"]
        )

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
            np.mean(
                client_rmse
            )
        ),
        "worst_client_rmse": float(
            np.max(
                client_rmse
            )
        ),
    }


def run_one(
    method_name,
    alpha,
    seed,
    cfg,
    train_dataset,
    test_dataset,
    device,
    run_dir,
):
    set_seed(
        seed
    )

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    client_sets = balanced_regime_partition(
        train_dataset,
        num_clients=cfg["data"]["num_clients"],
        alpha=float(alpha),
        seed=int(seed),
        affinity_weight=cfg["data"].get(
            "regime_affinity_weight",
            0.20,
        ),
    )

    splits = [
        split_subset_by_unit(
            subset,
            val_fraction=cfg["data"].get(
                "val_fraction",
                0.2,
            ),
            seed=int(seed) + i,
        )
        for i, subset in enumerate(
            client_sets
        )
    ]

    train_sets = [
        item[0]
        for item in splits
    ]

    val_sets = [
        item[1]
        for item in splits
    ]

    train_loaders = [
        make_loader(
            ds,
            cfg,
            shuffle=True,
        )
        for ds in train_sets
    ]

    val_loaders = [
        make_loader(
            ds,
            cfg,
            shuffle=False,
        )
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
    ).cpu()

    client_models = [
        build_model(
            cfg,
            method_name,
        ).to(
            device
        )
        for _ in train_sets
    ]

    is_scaffold = (
        method_name
        == "scaffold_transformer"
    )

    if is_scaffold:
        server_control = init_control(
            global_model
        )

        client_controls = [
            init_control(
                global_model
            )
            for _ in train_sets
        ]
    else:
        server_control = None
        client_controls = None

    best_rmse = None
    best_round = None
    best_state = None
    best_validation = None

    round_rows = []

    start = time.perf_counter()

    for round_idx in range(
        1,
        int(
            cfg["training"]["rounds"]
        )
        + 1,
    ):
        global_state = clone_cpu_state(
            global_model
        )

        client_states = []
        sample_counts = []
        local_losses = []

        control_deltas = []

        for cid, model in enumerate(
            client_models
        ):
            model.load_state_dict(
                global_state,
                strict=True,
            )

            if is_scaffold:
                (
                    local_loss,
                    local_state,
                    new_ci,
                    delta_ci,
                ) = train_scaffold_local(
                    model,
                    train_loaders[cid],
                    device,
                    cfg,
                    global_state,
                    server_control,
                    client_controls[cid],
                )

                client_controls[
                    cid
                ] = new_ci

                control_deltas.append(
                    delta_ci
                )

                client_states.append(
                    local_state
                )
            else:
                local_loss = train_fedavg_local(
                    model,
                    train_loaders[cid],
                    device,
                    cfg,
                )

                client_states.append(
                    clone_cpu_state(
                        model
                    )
                )

            local_losses.append(
                local_loss
            )

            sample_counts.append(
                len(
                    train_sets[cid]
                )
            )

        aggregated = weighted_average_states(
            client_states,
            sample_counts,
        )

        global_model.load_state_dict(
            aggregated,
            strict=True,
        )

        if is_scaffold:
            num_clients = float(
                len(
                    client_models
                )
            )

            for name in (
                server_control.keys()
            ):
                delta_mean = torch.zeros_like(
                    server_control[name]
                )

                for delta_ci in (
                    control_deltas
                ):
                    delta_mean.add_(
                        delta_ci[name],
                        alpha=(
                            1.0
                            / num_clients
                        ),
                    )

                server_control[
                    name
                ] = (
                    server_control[
                        name
                    ]
                    + delta_mean
                )

        for model in (
            client_models
        ):
            model.load_state_dict(
                aggregated,
                strict=True,
            )

        validation = evaluate_pooled(
            client_models,
            val_loaders,
            device,
            cfg,
        )

        current_rmse = float(
            validation[
                "pooled_rmse"
            ]
        )

        round_rows.append({
            "round": round_idx,
            "mean_train_loss": float(
                np.mean(
                    local_losses
                )
            ),
            **validation,
        })

        if (
            best_rmse is None
            or current_rmse
            < best_rmse
        ):
            best_rmse = current_rmse
            best_round = int(
                round_idx
            )
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
            == int(
                cfg[
                    "training"
                ][
                    "rounds"
                ]
            )
        ):
            print(
                f"    round "
                f"{round_idx:02d}/"
                f"{cfg['training']['rounds']} | "
                f"val RMSE="
                f"{current_rmse:.3f} | "
                f"best="
                f"{best_rmse:.3f}",
                flush=True,
            )

    train_seconds = (
        time.perf_counter()
        - start
    )

    test_model = build_model(
        cfg,
        method_name,
    )

    test_model.load_state_dict(
        best_state,
        strict=True,
    )

    test_model = test_model.to(
        device
    )

    test_result = predict(
        test_model,
        test_loader,
        device,
        cfg,
    )

    test_metrics = metric_dict(
        test_result["pred"],
        test_result["target"],
    )

    pd.DataFrame(
        round_rows
    ).to_csv(
        run_dir
        / "round_metrics.csv",
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
            "method_metadata": METHODS[
                method_name
            ],
            "config": cfg,
        },
        run_dir
        / "best_model.pt",
    )

    result = {
        "method": method_name,
        "display_name": METHODS[
            method_name
        ]["display_name"],
        "origin": METHODS[
            method_name
        ]["origin"],
        "baseline_type": METHODS[
            method_name
        ]["type"],
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
        run_dir
        / "result.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    return result


def build_tables(
    study_dir,
):
    rows = []

    for result_path in (
        study_dir.glob(
            "runs/*/*/*/result.json"
        )
    ):
        rows.append(
            json.loads(
                result_path.read_text(
                    encoding="utf-8"
                )
            )
        )

    raw = pd.DataFrame(
        rows
    )

    raw_path = (
        study_dir
        / "published_raw_results.csv"
    )

    raw.to_csv(
        raw_path,
        index=False,
    )

    if raw.empty:
        return (
            raw_path,
            None,
        )

    grouped_rows = []

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
            "display_name": (
                group[
                    "display_name"
                ].iloc[0]
            ),
            "origin": (
                group[
                    "origin"
                ].iloc[0]
            ),
            "baseline_type": (
                group[
                    "baseline_type"
                ].iloc[0]
            ),
            "alpha": float(
                alpha
            ),
            "n_seeds": int(
                len(
                    group
                )
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
                group[
                    metric
                ],
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
                if len(
                    values
                ) > 1
                else 0.0
            )

        grouped_rows.append(
            row
        )

    summary = pd.DataFrame(
        grouped_rows
    )

    summary_path = (
        study_dir
        / "published_mean_std.csv"
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
        default=(
            "configs/"
            "published_baselines_fd004.yaml"
        ),
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

    train_dataset = (
        load_processed_cmapss(
            cfg[
                "data"
            ][
                "train_path"
            ]
        )
    )

    test_dataset = (
        load_processed_cmapss(
            cfg[
                "data"
            ][
                "test_path"
            ]
        )
    )

    methods = (
        args.methods
        if args.methods
        else cfg[
            "study"
        ][
            "methods"
        ]
    )

    alphas = (
        args.alphas
        if args.alphas
        else cfg[
            "study"
        ][
            "alphas"
        ]
    )

    seeds = (
        args.seeds
        if args.seeds
        else cfg[
            "study"
        ][
            "seeds"
        ]
    )

    unknown = [
        method
        for method in methods
        if method not in METHODS
    ]

    if unknown:
        raise ValueError(
            f"Unknown methods: "
            f"{unknown}"
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
        stamp = (
            datetime.now()
            .strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        study_dir = (
            output_root
            / (
                "published_baselines_"
                f"FD004_{stamp}"
            )
        )

    study_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        output_root
        / "latest_published_baselines.txt"
    ).write_text(
        str(
            study_dir.resolve()
        ),
        encoding="utf-8",
    )

    total_jobs = (
        len(
            methods
        )
        * len(
            alphas
        )
        * len(
            seeds
        )
    )

    job_idx = 0
    start = time.perf_counter()

    for alpha in alphas:
        for seed in seeds:
            for method_name in methods:
                job_idx += 1

                run_dir = (
                    study_dir
                    / "runs"
                    / f"alpha_{alpha}"
                    / f"seed_{seed}"
                    / method_name
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
                        f"[{job_idx}/"
                        f"{total_jobs}] "
                        f"skip "
                        f"{method_name}, "
                        f"alpha={alpha}, "
                        f"seed={seed}"
                    )
                    continue

                print(
                    f"\n[{job_idx}/"
                    f"{total_jobs}] "
                    f"{method_name} | "
                    f"alpha={alpha} | "
                    f"seed={seed}"
                )

                result = run_one(
                    method_name,
                    alpha,
                    seed,
                    cfg,
                    train_dataset,
                    test_dataset,
                    device,
                    run_dir,
                )

                print(
                    f"  done | "
                    f"val RMSE="
                    f"{result['val_rmse']:.3f} | "
                    f"test RMSE="
                    f"{result['test_rmse']:.3f} | "
                    f"time="
                    f"{result['train_seconds']:.1f}s"
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

    (
        study_dir
        / "study_summary.json"
    ).write_text(
        json.dumps(
            {
                "study_dir": str(
                    study_dir
                ),
                "methods": methods,
                "alphas": alphas,
                "seeds": seeds,
                "total_jobs": (
                    total_jobs
                ),
                "raw_results": str(
                    raw_path
                ),
                "mean_std_results": (
                    str(
                        summary_path
                    )
                    if summary_path
                    else None
                ),
                "elapsed_seconds": (
                    float(
                        elapsed
                    )
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nPublished-baseline "
        "study complete."
    )

    print(
        f"Raw results: "
        f"{raw_path}"
    )

    print(
        f"Mean/std: "
        f"{summary_path}"
    )

    print(
        f"Elapsed: "
        f"{elapsed / 60.0:.2f} min"
    )


if __name__ == "__main__":
    main()
