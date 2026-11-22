import argparse
import copy
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twinfed_moe.data.cmapss import load_processed_cmapss
from twinfed_moe.data.partition import (
    balanced_regime_partition,
    split_subset_by_unit,
)
from twinfed_moe.federated.benchmark_aggregation import (
    aggregate_fedavg,
    aggregate_expertwise,
)
from twinfed_moe.models.benchmark_model import (
    BenchmarkRULModel,
)
from twinfed_moe.training.metrics import (
    metric_dict,
)
from twinfed_moe.training.trainer import (
    train_local,
    predict,
)
from twinfed_moe.utils.config import (
    load_config,
)
from twinfed_moe.utils.device import (
    resolve_device,
)
from twinfed_moe.utils.seed import (
    set_seed,
)


METHODS = {
    "centralized_transformer": {
        "category": "reference",
        "mode": "centralized",
        "use_digital_twin": False,
        "use_moe": False,
        "use_personalization": False,
        "router_mode": "temporal",
        "aggregation": None,
        "prox_mu": 0.0,
        "lambda_dt": 0.0,
        "lambda_bal": 0.0,
        "lambda_tgc": 0.0,
        "description": "Centralized temporal Transformer reference.",
    },

    "local_only_transformer": {
        "category": "reference",
        "mode": "local_only",
        "use_digital_twin": False,
        "use_moe": False,
        "use_personalization": False,
        "router_mode": "temporal",
        "aggregation": None,
        "prox_mu": 0.0,
        "lambda_dt": 0.0,
        "lambda_bal": 0.0,
        "lambda_tgc": 0.0,
        "description": "Independent local Transformer models with no federation.",
    },

    "fedavg_transformer": {
        "category": "baseline",
        "mode": "federated",
        "use_digital_twin": False,
        "use_moe": False,
        "use_personalization": False,
        "router_mode": "temporal",
        "aggregation": "fedavg",
        "prox_mu": 0.0,
        "lambda_dt": 0.0,
        "lambda_bal": 0.0,
        "lambda_tgc": 0.0,
        "description": "FedAvg with a temporal Transformer backbone.",
    },

    "fedprox_transformer": {
        "category": "baseline",
        "mode": "federated",
        "use_digital_twin": False,
        "use_moe": False,
        "use_personalization": False,
        "router_mode": "temporal",
        "aggregation": "fedavg",
        "prox_mu": 0.001,
        "lambda_dt": 0.0,
        "lambda_bal": 0.0,
        "lambda_tgc": 0.0,
        "description": "FedProx with a temporal Transformer backbone.",
    },

    "fedper_transformer": {
        "category": "baseline",
        "mode": "federated",
        "use_digital_twin": False,
        "use_moe": False,
        "use_personalization": True,
        "router_mode": "temporal",
        "aggregation": "fedavg",
        "prox_mu": 0.0,
        "lambda_dt": 0.0,
        "lambda_bal": 0.0,
        "lambda_tgc": 0.0,
        "description": "FedPer-style Transformer with local personalization adapter.",
    },

    "fedavg_moe": {
        "category": "baseline",
        "mode": "federated",
        "use_digital_twin": False,
        "use_moe": True,
        "use_personalization": False,
        "router_mode": "temporal",
        "aggregation": "fedavg",
        "prox_mu": 0.0,
        "lambda_dt": 0.0,
        "lambda_bal": 0.01,
        "lambda_tgc": 0.0,
        "description": "Sparse temporal MoE trained with ordinary FedAvg.",
    },

    "twinfed_moe": {
        "category": "proposed",
        "mode": "federated",
        "use_digital_twin": True,
        "use_moe": True,
        "use_personalization": True,
        "router_mode": "twin",
        "aggregation": "expertwise",
        "prox_mu": 0.001,
        "lambda_dt": 0.10,
        "lambda_bal": 0.01,
        "lambda_tgc": 0.01,
        "description": "Full TwinFed-MoE.",
    },

    "ablation_no_dt": {
        "category": "ablation",
        "mode": "federated",
        "use_digital_twin": False,
        "use_moe": True,
        "use_personalization": True,
        "router_mode": "temporal",
        "aggregation": "expertwise",
        "prox_mu": 0.001,
        "lambda_dt": 0.0,
        "lambda_bal": 0.01,
        "lambda_tgc": 0.0,
        "description": "TwinFed-MoE without the Digital Twin state.",
    },

    "ablation_no_moe": {
        "category": "ablation",
        "mode": "federated",
        "use_digital_twin": True,
        "use_moe": False,
        "use_personalization": True,
        "router_mode": "twin",
        "aggregation": "fedavg",
        "prox_mu": 0.001,
        "lambda_dt": 0.10,
        "lambda_bal": 0.0,
        "lambda_tgc": 0.0,
        "description": "Digital Twin + Transformer + personalization, without MoE.",
    },

    "ablation_no_personalization": {
        "category": "ablation",
        "mode": "federated",
        "use_digital_twin": True,
        "use_moe": True,
        "use_personalization": False,
        "router_mode": "twin",
        "aggregation": "expertwise",
        "prox_mu": 0.001,
        "lambda_dt": 0.10,
        "lambda_bal": 0.01,
        "lambda_tgc": 0.01,
        "description": "TwinFed-MoE without the local personalization adapter.",
    },

    "ablation_no_tgc": {
        "category": "ablation",
        "mode": "federated",
        "use_digital_twin": True,
        "use_moe": True,
        "use_personalization": True,
        "router_mode": "twin",
        "aggregation": "expertwise",
        "prox_mu": 0.001,
        "lambda_dt": 0.10,
        "lambda_bal": 0.01,
        "lambda_tgc": 0.0,
        "description": "TwinFed-MoE without Twin-Guided Consistency loss.",
    },

    "ablation_fedavg_aggregation": {
        "category": "ablation",
        "mode": "federated",
        "use_digital_twin": True,
        "use_moe": True,
        "use_personalization": True,
        "router_mode": "twin",
        "aggregation": "fedavg",
        "prox_mu": 0.001,
        "lambda_dt": 0.10,
        "lambda_bal": 0.01,
        "lambda_tgc": 0.01,
        "description": "Full architecture with ordinary FedAvg instead of expert-wise aggregation.",
    },
}


def clone_cpu_state(model):
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


def load_shared_preserve_adapter(
    model,
    global_state,
):
    current = model.state_dict()

    merged = {}

    for name, tensor in global_state.items():
        if (
            name.startswith(
                "personal_adapter."
            )
            and name in current
        ):
            merged[name] = current[name]
        else:
            merged[name] = tensor

    model.load_state_dict(
        merged,
        strict=True,
    )


def model_factory(
    base_cfg,
    method_spec,
    twin_aux_dim,
):
    m = base_cfg["model"]

    return BenchmarkRULModel(
        num_sensors=base_cfg["data"]["num_sensors"],
        twin_aux_dim=twin_aux_dim,
        seq_len=base_cfg["data"]["seq_len"],
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
        use_digital_twin=method_spec["use_digital_twin"],
        use_moe=method_spec["use_moe"],
        use_personalization=method_spec["use_personalization"],
        router_mode=method_spec["router_mode"],
    )


def method_cfg(
    base_cfg,
    method_spec,
):
    cfg = copy.deepcopy(
        base_cfg
    )

    cfg["training"]["prox_mu"] = float(
        method_spec["prox_mu"]
    )

    cfg["training"]["lambda_dt"] = float(
        method_spec["lambda_dt"]
    )

    cfg["training"]["lambda_bal"] = float(
        method_spec["lambda_bal"]
    )

    cfg["training"]["lambda_tgc"] = float(
        method_spec["lambda_tgc"]
    )

    return cfg


def make_loader(
    dataset,
    cfg,
    shuffle,
):
    return DataLoader(
        dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
    )


def evaluate_loader(
    model,
    loader,
    device,
    cfg,
    use_personal_adapter,
):
    result = predict(
        model,
        loader,
        device,
        cfg,
        use_personal_adapter=use_personal_adapter,
        collect_routing=False,
    )

    return (
        metric_dict(
            result["pred"],
            result["target"],
        ),
        result["pred"],
        result["target"],
    )


def official_test(
    model,
    test_loader,
    device,
    cfg,
):
    metrics, pred, target = evaluate_loader(
        model,
        test_loader,
        device,
        cfg,
        use_personal_adapter=False,
    )

    return metrics


def count_parameters(
    model,
):
    return int(
        sum(
            p.numel()
            for p in model.parameters()
        )
    )


def pooled_client_validation(
    client_models,
    val_loaders,
    device,
    cfg,
    use_personalization,
):
    personal_preds = []
    personal_targets = []

    shared_preds = []
    shared_targets = []

    per_client = []

    for model, loader in zip(
        client_models,
        val_loaders,
    ):
        personal_metrics, pred, target = (
            evaluate_loader(
                model,
                loader,
                device,
                cfg,
                use_personal_adapter=(
                    use_personalization
                ),
            )
        )

        per_client.append(
            personal_metrics
        )

        personal_preds.append(
            pred
        )

        personal_targets.append(
            target
        )

        _, shared_pred, shared_target = (
            evaluate_loader(
                model,
                loader,
                device,
                cfg,
                use_personal_adapter=False,
            )
        )

        shared_preds.append(
            shared_pred
        )

        shared_targets.append(
            shared_target
        )

    pooled_pred = torch.cat(
        personal_preds
    )

    pooled_target = torch.cat(
        personal_targets
    )

    shared_pred = torch.cat(
        shared_preds
    )

    shared_target = torch.cat(
        shared_targets
    )

    pooled = metric_dict(
        pooled_pred,
        pooled_target,
    )

    shared = metric_dict(
        shared_pred,
        shared_target,
    )

    return {
        "pooled_rmse": pooled["rmse"],
        "pooled_mae": pooled["mae"],
        "pooled_r2": pooled["r2"],
        "pooled_nasa_score": pooled["nasa_score"],

        "mean_client_rmse": float(
            np.mean(
                [
                    x["rmse"]
                    for x in per_client
                ]
            )
        ),

        "mean_client_mae": float(
            np.mean(
                [
                    x["mae"]
                    for x in per_client
                ]
            )
        ),

        "worst_client_rmse": float(
            np.max(
                [
                    x["rmse"]
                    for x in per_client
                ]
            )
        ),

        "shared_pooled_rmse": shared["rmse"],
        "shared_pooled_mae": shared["mae"],
        "shared_pooled_r2": shared["r2"],
        "shared_pooled_nasa_score": shared["nasa_score"],

        "personalization_gain_rmse": float(
            shared["rmse"]
            - pooled["rmse"]
        ),
    }


def aggregate(
    method_spec,
    states,
    sample_counts,
    utilizations,
):
    if (
        method_spec[
            "aggregation"
        ]
        == "expertwise"
    ):
        return aggregate_expertwise(
            states,
            sample_counts,
            utilizations,
        )

    return aggregate_fedavg(
        states,
        sample_counts,
    )


def run_federated(
    method_name,
    method_spec,
    cfg,
    train_sets,
    val_sets,
    test_loader,
    twin_aux_dim,
    device,
    method_dir,
):
    print(
        f"\n=== {method_name} ==="
    )

    global_model = model_factory(
        cfg,
        method_spec,
        twin_aux_dim,
    ).cpu()

    client_models = [
        model_factory(
            cfg,
            method_spec,
            twin_aux_dim,
        ).to(device)
        for _ in train_sets
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

    best_rmse = None
    best_round = None
    best_global = None
    best_adapters = None
    best_validation = None

    round_rows = []

    start = time.perf_counter()

    total_rounds = int(
        cfg["training"]["rounds"]
    )

    participation = float(
        cfg["federated"][
            "participation_rate"
        ]
    )

    for round_idx in range(
        1,
        total_rounds + 1,
    ):
        global_state = clone_cpu_state(
            global_model
        )

        n_selected = max(
            1,
            int(
                round(
                    len(
                        client_models
                    )
                    * participation
                )
            ),
        )

        selected_ids = random.sample(
            list(
                range(
                    len(
                        client_models
                    )
                )
            ),
            n_selected,
        )

        states = []
        sample_counts = []
        utilizations = []
        local_losses = []

        for cid in selected_ids:
            model = client_models[
                cid
            ]

            load_shared_preserve_adapter(
                model,
                global_state,
            )

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg["training"]["lr"],
                weight_decay=cfg["training"]["weight_decay"],
            )

            train_result = train_local(
                model,
                train_loaders[cid],
                optimizer,
                device,
                cfg,
                global_state=global_state,
            )

            states.append(
                clone_cpu_state(
                    model
                )
            )

            sample_counts.append(
                len(
                    train_sets[cid]
                )
            )

            utilizations.append(
                train_result[
                    "expert_utilization"
                ]
            )

            local_losses.append(
                train_result[
                    "loss"
                ]
            )

        aggregated = aggregate(
            method_spec,
            states,
            sample_counts,
            utilizations,
        )

        global_model.load_state_dict(
            aggregated,
            strict=True,
        )

        for model in client_models:
            load_shared_preserve_adapter(
                model,
                aggregated,
            )

        validation = pooled_client_validation(
            client_models,
            val_loaders,
            device,
            cfg,
            use_personalization=(
                method_spec[
                    "use_personalization"
                ]
            ),
        )

        row = {
            "round": round_idx,
            "mean_train_loss": float(
                np.mean(
                    local_losses
                )
            ),
            **validation,
        }

        round_rows.append(
            row
        )

        if (
            best_rmse is None
            or validation[
                "pooled_rmse"
            ] < best_rmse
        ):
            best_rmse = float(
                validation[
                    "pooled_rmse"
                ]
            )

            best_round = int(
                round_idx
            )

            best_global = clone_cpu_state(
                global_model
            )

            best_adapters = {
                cid: {
                    k: v.detach()
                    .cpu()
                    .clone()
                    for k, v in model.state_dict().items()
                    if k.startswith(
                        "personal_adapter."
                    )
                }
                for cid, model
                in enumerate(
                    client_models
                )
            }

            best_validation = dict(
                validation
            )

        if (
            round_idx == 1
            or round_idx % 5 == 0
            or round_idx
            == total_rounds
        ):
            print(
                f"  round {round_idx:02d}/"
                f"{total_rounds} | "
                f"val RMSE="
                f"{validation['pooled_rmse']:.3f} | "
                f"best="
                f"{best_rmse:.3f}",
                flush=True,
            )

    train_seconds = (
        time.perf_counter()
        - start
    )

    # Restore best shared checkpoint.
    global_model.load_state_dict(
        best_global,
        strict=True,
    )

    # Restore best shared params and client-local adapters.
    for cid, model in enumerate(
        client_models
    ):
        load_shared_preserve_adapter(
            model,
            best_global,
        )

        current = model.state_dict()

        for name, tensor in (
            best_adapters[
                cid
            ].items()
        ):
            if name in current:
                current[name] = tensor

        model.load_state_dict(
            current,
            strict=True,
        )

    final_validation = (
        pooled_client_validation(
            client_models,
            val_loaders,
            device,
            cfg,
            use_personalization=(
                method_spec[
                    "use_personalization"
                ]
            ),
        )
    )

    test_model = model_factory(
        cfg,
        method_spec,
        twin_aux_dim,
    )

    test_model.load_state_dict(
        best_global,
        strict=True,
    )

    test_model = test_model.to(
        device
    )

    test_metrics = official_test(
        test_model,
        test_loader,
        device,
        cfg,
    )

    pd.DataFrame(
        round_rows
    ).to_csv(
        method_dir
        / "round_metrics.csv",
        index=False,
    )

    torch.save(
        {
            "method": method_name,
            "method_spec": method_spec,
            "best_round": best_round,
            "validation": final_validation,
            "global_model_state": best_global,
            "client_adapter_states": best_adapters,
            "config": cfg,
        },
        method_dir
        / "best_model.pt",
    )

    return {
        "method": method_name,
        "category": method_spec[
            "category"
        ],
        "description": method_spec[
            "description"
        ],
        "mode": "federated",
        "aggregation": method_spec[
            "aggregation"
        ],
        "best_round_or_epoch": best_round,
        "val_rmse": final_validation[
            "pooled_rmse"
        ],
        "val_mae": final_validation[
            "pooled_mae"
        ],
        "val_r2": final_validation[
            "pooled_r2"
        ],
        "val_nasa": final_validation[
            "pooled_nasa_score"
        ],
        "mean_client_rmse": final_validation[
            "mean_client_rmse"
        ],
        "worst_client_rmse": final_validation[
            "worst_client_rmse"
        ],
        "shared_val_rmse": final_validation[
            "shared_pooled_rmse"
        ],
        "personalization_gain_rmse": final_validation[
            "personalization_gain_rmse"
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
        "train_seconds": train_seconds,
        "parameters": count_parameters(
            test_model
        ),
    }


def run_centralized(
    method_name,
    method_spec,
    cfg,
    train_sets,
    val_sets,
    test_loader,
    twin_aux_dim,
    device,
    method_dir,
):
    print(
        f"\n=== {method_name} ==="
    )

    train_dataset = ConcatDataset(
        train_sets
    )

    val_dataset = ConcatDataset(
        val_sets
    )

    train_loader = make_loader(
        train_dataset,
        cfg,
        shuffle=True,
    )

    val_loader = make_loader(
        val_dataset,
        cfg,
        shuffle=False,
    )

    model = model_factory(
        cfg,
        method_spec,
        twin_aux_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    epoch_cfg = copy.deepcopy(
        cfg
    )

    epoch_cfg[
        "training"
    ][
        "local_epochs"
    ] = 1

    max_epochs = int(
        cfg["training"].get(
            "centralized_epochs",
            40,
        )
    )

    patience = int(
        cfg["training"].get(
            "early_stopping_patience",
            8,
        )
    )

    best_rmse = None
    best_epoch = None
    best_state = None
    best_val = None
    no_improve = 0
    rows = []

    start = time.perf_counter()

    for epoch in range(
        1,
        max_epochs + 1,
    ):
        train_result = train_local(
            model,
            train_loader,
            optimizer,
            device,
            epoch_cfg,
            global_state=None,
        )

        val_metrics, _, _ = (
            evaluate_loader(
                model,
                val_loader,
                device,
                cfg,
                use_personal_adapter=False,
            )
        )

        rows.append({
            "epoch": epoch,
            "train_loss": train_result[
                "loss"
            ],
            **{
                f"val_{k}": v
                for k, v
                in val_metrics.items()
            },
        })

        if (
            best_rmse is None
            or val_metrics[
                "rmse"
            ] < best_rmse
        ):
            best_rmse = float(
                val_metrics[
                    "rmse"
                ]
            )

            best_epoch = int(
                epoch
            )

            best_state = clone_cpu_state(
                model
            )

            best_val = dict(
                val_metrics
            )

            no_improve = 0
        else:
            no_improve += 1

        if (
            epoch == 1
            or epoch % 5 == 0
        ):
            print(
                f"  epoch {epoch:02d}/"
                f"{max_epochs} | "
                f"val RMSE="
                f"{val_metrics['rmse']:.3f} | "
                f"best="
                f"{best_rmse:.3f}",
                flush=True,
            )

        if no_improve >= patience:
            print(
                f"  early stop at epoch "
                f"{epoch}",
                flush=True,
            )

            break

    train_seconds = (
        time.perf_counter()
        - start
    )

    model.load_state_dict(
        best_state,
        strict=True,
    )

    test_metrics = official_test(
        model,
        test_loader,
        device,
        cfg,
    )

    pd.DataFrame(
        rows
    ).to_csv(
        method_dir
        / "epoch_metrics.csv",
        index=False,
    )

    torch.save(
        {
            "method": method_name,
            "best_epoch": best_epoch,
            "validation": best_val,
            "global_model_state": best_state,
            "config": cfg,
        },
        method_dir
        / "best_model.pt",
    )

    return {
        "method": method_name,
        "category": method_spec[
            "category"
        ],
        "description": method_spec[
            "description"
        ],
        "mode": "centralized",
        "aggregation": "",
        "best_round_or_epoch": best_epoch,
        "val_rmse": best_val[
            "rmse"
        ],
        "val_mae": best_val[
            "mae"
        ],
        "val_r2": best_val[
            "r2"
        ],
        "val_nasa": best_val[
            "nasa_score"
        ],
        "mean_client_rmse": None,
        "worst_client_rmse": None,
        "shared_val_rmse": best_val[
            "rmse"
        ],
        "personalization_gain_rmse": 0.0,
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
        "train_seconds": train_seconds,
        "parameters": count_parameters(
            model
        ),
    }


def run_local_only(
    method_name,
    method_spec,
    cfg,
    train_sets,
    val_sets,
    twin_aux_dim,
    device,
    method_dir,
):
    print(
        f"\n=== {method_name} ==="
    )

    max_epochs = int(
        cfg["training"].get(
            "local_only_epochs",
            40,
        )
    )

    patience = int(
        cfg["training"].get(
            "early_stopping_patience",
            8,
        )
    )

    epoch_cfg = copy.deepcopy(
        cfg
    )

    epoch_cfg[
        "training"
    ][
        "local_epochs"
    ] = 1

    pooled_preds = []
    pooled_targets = []
    client_metrics = []

    total_seconds = 0.0
    best_epochs = []
    parameter_count = None

    for cid, (
        train_ds,
        val_ds,
    ) in enumerate(
        zip(
            train_sets,
            val_sets,
        )
    ):
        model = model_factory(
            cfg,
            method_spec,
            twin_aux_dim,
        ).to(device)

        if parameter_count is None:
            parameter_count = (
                count_parameters(
                    model
                )
            )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["training"]["lr"],
            weight_decay=cfg["training"]["weight_decay"],
        )

        train_loader = make_loader(
            train_ds,
            cfg,
            shuffle=True,
        )

        val_loader = make_loader(
            val_ds,
            cfg,
            shuffle=False,
        )

        best_rmse = None
        best_state = None
        best_epoch = None
        no_improve = 0

        start = time.perf_counter()

        for epoch in range(
            1,
            max_epochs + 1,
        ):
            train_local(
                model,
                train_loader,
                optimizer,
                device,
                epoch_cfg,
                global_state=None,
            )

            metrics, _, _ = (
                evaluate_loader(
                    model,
                    val_loader,
                    device,
                    cfg,
                    use_personal_adapter=False,
                )
            )

            if (
                best_rmse is None
                or metrics[
                    "rmse"
                ] < best_rmse
            ):
                best_rmse = float(
                    metrics[
                        "rmse"
                    ]
                )

                best_state = (
                    clone_cpu_state(
                        model
                    )
                )

                best_epoch = int(
                    epoch
                )

                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                break

        total_seconds += (
            time.perf_counter()
            - start
        )

        model.load_state_dict(
            best_state,
            strict=True,
        )

        metrics, pred, target = (
            evaluate_loader(
                model,
                val_loader,
                device,
                cfg,
                use_personal_adapter=False,
            )
        )

        pooled_preds.append(
            pred
        )

        pooled_targets.append(
            target
        )

        client_metrics.append(
            metrics
        )

        best_epochs.append(
            best_epoch
        )

        print(
            f"  client {cid}: "
            f"best epoch={best_epoch}, "
            f"RMSE="
            f"{metrics['rmse']:.3f}",
            flush=True,
        )

    pooled = metric_dict(
        torch.cat(
            pooled_preds
        ),
        torch.cat(
            pooled_targets
        ),
    )

    client_rows = []

    for cid, metrics in enumerate(
        client_metrics
    ):
        client_rows.append({
            "client_id": cid,
            "best_epoch": best_epochs[
                cid
            ],
            **metrics,
        })

    pd.DataFrame(
        client_rows
    ).to_csv(
        method_dir
        / "client_metrics.csv",
        index=False,
    )

    # Deliberately no official-test number:
    # there is no single global/local-only model for an unseen engine.
    return {
        "method": method_name,
        "category": method_spec[
            "category"
        ],
        "description": method_spec[
            "description"
        ],
        "mode": "local_only",
        "aggregation": "",
        "best_round_or_epoch": float(
            np.mean(
                best_epochs
            )
        ),
        "val_rmse": pooled[
            "rmse"
        ],
        "val_mae": pooled[
            "mae"
        ],
        "val_r2": pooled[
            "r2"
        ],
        "val_nasa": pooled[
            "nasa_score"
        ],
        "mean_client_rmse": float(
            np.mean(
                [
                    x["rmse"]
                    for x in client_metrics
                ]
            )
        ),
        "worst_client_rmse": float(
            np.max(
                [
                    x["rmse"]
                    for x in client_metrics
                ]
            )
        ),
        "shared_val_rmse": pooled[
            "rmse"
        ],
        "personalization_gain_rmse": 0.0,
        "test_rmse": None,
        "test_mae": None,
        "test_r2": None,
        "test_nasa": None,
        "train_seconds": total_seconds,
        "parameters": parameter_count,
    }


def save_result(
    result,
    method_dir,
):
    (
        method_dir
        / "result.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )


def build_summary(
    suite_dir,
):
    result_files = sorted(
        suite_dir.glob(
            "*/result.json"
        )
    )

    rows = []

    for path in result_files:
        rows.append(
            json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        )

    df = pd.DataFrame(
        rows
    )

    if len(df):
        order = {
            name: i
            for i, name in enumerate(
                METHODS.keys()
            )
        }

        df["_order"] = (
            df["method"].map(
                order
            )
        )

        df = (
            df.sort_values(
                "_order"
            )
            .drop(
                columns=[
                    "_order"
                ]
            )
        )

    out = (
        suite_dir
        / "benchmark_results.csv"
    )

    df.to_csv(
        out,
        index=False,
    )

    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=(
            "configs/"
            "benchmark_fd004.yaml"
        ),
    )

    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help=(
            "Optional subset of method names. "
            "Default: methods from benchmark config."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional seed override. "
            "Default: seed from benchmark config."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip methods that already have result.json "
            "inside the suite directory."
        ),
    )

    parser.add_argument(
        "--suite-dir",
        default=None,
        help=(
            "Existing benchmark directory for --resume. "
            "If omitted, a new timestamped directory is created."
        ),
    )

    args = parser.parse_args()

    cfg = load_config(
        args.config
    )

    if args.seed is not None:
        cfg["seed"] = int(
            args.seed
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
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    dataset = load_processed_cmapss(
        cfg["data"]["train_path"]
    )

    test_dataset = load_processed_cmapss(
        cfg["data"]["test_path"]
    )

    twin_aux_dim = int(
        dataset.twin_aux.shape[-1]
    )

    client_sets = (
        balanced_regime_partition(
            dataset,
            num_clients=cfg["data"]["num_clients"],
            alpha=cfg["data"]["non_iid_alpha"],
            seed=cfg["seed"],
            affinity_weight=cfg["data"].get(
                "regime_affinity_weight",
                0.20,
            ),
        )
    )

    splits = [
        split_subset_by_unit(
            subset,
            val_fraction=cfg["data"].get(
                "val_fraction",
                0.2,
            ),
            seed=cfg["seed"] + i,
        )
        for i, subset
        in enumerate(
            client_sets
        )
    ]

    train_sets = [
        x[0]
        for x in splits
    ]

    val_sets = [
        x[1]
        for x in splits
    ]

    print(
        f"FD004 train windows: "
        f"{len(dataset):,}"
    )

    print(
        "Client train windows:",
        [
            len(x)
            for x in train_sets
        ],
    )

    print(
        "Client validation windows:",
        [
            len(x)
            for x in val_sets
        ],
    )

    test_loader = make_loader(
        test_dataset,
        cfg,
        shuffle=False,
    )

    requested_methods = (
        args.methods
        if args.methods
        else cfg.get(
            "benchmark",
            {},
        ).get(
            "methods",
            list(
                METHODS.keys()
            ),
        )
    )

    unknown = [
        m
        for m in requested_methods
        if m not in METHODS
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

    if args.suite_dir:
        suite_dir = Path(
            args.suite_dir
        )
    else:
        stamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        suite_dir = (
            output_root
            / (
                f"benchmark_FD004_"
                f"{stamp}"
            )
        )

    suite_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        output_root
        / "latest_benchmark.txt"
    ).write_text(
        str(
            suite_dir.resolve()
        ),
        encoding="utf-8",
    )

    (
        suite_dir
        / "method_definitions.json"
    ).write_text(
        json.dumps(
            METHODS,
            indent=2,
        ),
        encoding="utf-8",
    )

    suite_start = (
        time.perf_counter()
    )

    for method_name in (
        requested_methods
    ):
        method_spec = METHODS[
            method_name
        ]

        method_dir = (
            suite_dir
            / method_name
        )

        method_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        result_path = (
            method_dir
            / "result.json"
        )

        if (
            args.resume
            and result_path.exists()
        ):
            print(
                f"\nSkipping completed: "
                f"{method_name}"
            )

            continue

        # Reset all RNGs before every method.
        # This makes comparisons deterministic and reduces nuisance variation.
        set_seed(
            cfg["seed"]
        )

        random.seed(
            cfg["seed"]
        )

        np.random.seed(
            cfg["seed"]
        )

        torch.manual_seed(
            cfg["seed"]
        )

        this_cfg = method_cfg(
            cfg,
            method_spec,
        )

        mode = method_spec[
            "mode"
        ]

        if mode == "federated":
            result = run_federated(
                method_name,
                method_spec,
                this_cfg,
                train_sets,
                val_sets,
                test_loader,
                twin_aux_dim,
                device,
                method_dir,
            )

        elif mode == "centralized":
            result = run_centralized(
                method_name,
                method_spec,
                this_cfg,
                train_sets,
                val_sets,
                test_loader,
                twin_aux_dim,
                device,
                method_dir,
            )

        elif mode == "local_only":
            result = run_local_only(
                method_name,
                method_spec,
                this_cfg,
                train_sets,
                val_sets,
                twin_aux_dim,
                device,
                method_dir,
            )

        else:
            raise ValueError(
                mode
            )

        save_result(
            result,
            method_dir,
        )

        comparison_path = build_summary(
            suite_dir
        )

        print(
            f"  completed "
            f"{method_name} | "
            f"val RMSE="
            f"{result['val_rmse']:.3f} | "
            f"test RMSE="
            f"{result['test_rmse']}"
        )

        print(
            f"  current table: "
            f"{comparison_path}"
        )

    total_seconds = (
        time.perf_counter()
        - suite_start
    )

    comparison_path = build_summary(
        suite_dir
    )

    summary = {
        "suite_dir": str(
            suite_dir
        ),
        "seed": int(
            cfg["seed"]
        ),
        "methods_requested": (
            requested_methods
        ),
        "total_seconds_this_invocation": float(
            total_seconds
        ),
        "results_csv": str(
            comparison_path
        ),
    }

    (
        suite_dir
        / "suite_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\nBenchmark suite complete."
    )

    print(
        f"Results: "
        f"{comparison_path}"
    )

    print(
        f"Elapsed this invocation: "
        f"{total_seconds / 60.0:.2f} min"
    )


if __name__ == "__main__":
    main()
