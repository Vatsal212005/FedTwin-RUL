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
from twinfed_moe.data.partition import (
    balanced_regime_partition,
    split_subset_by_unit,
)
from twinfed_moe.federated.v2_aggregation import (
    aggregate_v2,
)
from twinfed_moe.models.v2_model import (
    TwinFedMoEV2,
)
from twinfed_moe.training.metrics import (
    metric_dict,
)
from twinfed_moe.training.v2_trainer import (
    is_local_parameter,
    metrics_from_prediction,
    predict_v2,
    train_local_v2,
)
from twinfed_moe.utils.config import load_config
from twinfed_moe.utils.device import resolve_device
from twinfed_moe.utils.seed import set_seed


METHODS = {
    "fedper_residual_v2": {
        "category": "baseline_v2",
        "use_digital_twin": False,
        "use_moe": False,
        "use_personalization": True,
        "router_mode": "temporal",
        "aggregation": "fedavg",
        "prox_mu": 0.0,
        "lambda_dt": 0.0,
        "lambda_bal": 0.0,
        "lambda_tgc": 0.0,
        "description": (
            "FedPer-style Transformer with directly supervised shared head "
            "and bounded client-local residual personalization."
        ),
    },

    "twinfed_moe_v2": {
        "category": "proposed_v2",
        "use_digital_twin": True,
        "use_moe": True,
        "use_personalization": True,
        "router_mode": "twin",
        "aggregation": "expertwise",
        "prox_mu": 0.001,
        "lambda_dt": 0.10,
        "lambda_bal": 0.01,
        "lambda_tgc": 0.01,
        "description": (
            "TwinFed-MoE v2 with a directly supervised global RUL head "
            "and bounded local residual personalization."
        ),
    },

    "v2_no_dt": {
        "category": "ablation_v2",
        "use_digital_twin": False,
        "use_moe": True,
        "use_personalization": True,
        "router_mode": "temporal",
        "aggregation": "expertwise",
        "prox_mu": 0.001,
        "lambda_dt": 0.0,
        "lambda_bal": 0.01,
        "lambda_tgc": 0.0,
        "description": "TwinFed-MoE v2 without Digital Twin state.",
    },

    "v2_no_moe": {
        "category": "ablation_v2",
        "use_digital_twin": True,
        "use_moe": False,
        "use_personalization": True,
        "router_mode": "twin",
        "aggregation": "fedavg",
        "prox_mu": 0.001,
        "lambda_dt": 0.10,
        "lambda_bal": 0.0,
        "lambda_tgc": 0.0,
        "description": "TwinFed-MoE v2 without the sparse MoE.",
    },

    "v2_no_personalization": {
        "category": "ablation_v2",
        "use_digital_twin": True,
        "use_moe": True,
        "use_personalization": False,
        "router_mode": "twin",
        "aggregation": "expertwise",
        "prox_mu": 0.001,
        "lambda_dt": 0.10,
        "lambda_bal": 0.01,
        "lambda_tgc": 0.01,
        "description": "TwinFed-MoE v2 without local residual personalization.",
    },

    "v2_no_tgc": {
        "category": "ablation_v2",
        "use_digital_twin": True,
        "use_moe": True,
        "use_personalization": True,
        "router_mode": "twin",
        "aggregation": "expertwise",
        "prox_mu": 0.001,
        "lambda_dt": 0.10,
        "lambda_bal": 0.01,
        "lambda_tgc": 0.0,
        "description": "TwinFed-MoE v2 without Twin-Guided Consistency loss.",
    },

    "v2_fedavg_aggregation": {
        "category": "ablation_v2",
        "use_digital_twin": True,
        "use_moe": True,
        "use_personalization": True,
        "router_mode": "twin",
        "aggregation": "fedavg",
        "prox_mu": 0.001,
        "lambda_dt": 0.10,
        "lambda_bal": 0.01,
        "lambda_tgc": 0.01,
        "description": (
            "TwinFed-MoE v2 with ordinary FedAvg instead of "
            "expert-utilization-aware aggregation."
        ),
    },
}


def clone_cpu_state(
    model,
):
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


def local_state(
    model,
):
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if is_local_parameter(k)
    }


def load_global_preserve_local(
    model,
    global_state,
):
    current = model.state_dict()
    merged = {}

    for name, tensor in (
        global_state.items()
    ):
        if (
            is_local_parameter(name)
            and name in current
        ):
            merged[name] = current[name]
        else:
            merged[name] = tensor

    model.load_state_dict(
        merged,
        strict=True,
    )


def restore_local_state(
    model,
    saved_local,
):
    state = model.state_dict()

    for name, tensor in (
        saved_local.items()
    ):
        if name in state:
            state[name] = tensor

    model.load_state_dict(
        state,
        strict=True,
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


def method_cfg(
    base_cfg,
    spec,
):
    cfg = copy.deepcopy(
        base_cfg
    )

    for key in (
        "prox_mu",
        "lambda_dt",
        "lambda_bal",
        "lambda_tgc",
    ):
        cfg["training"][key] = float(
            spec[key]
        )

    return cfg


def build_model(
    cfg,
    spec,
    twin_aux_dim,
):
    m = cfg["model"]

    return TwinFedMoEV2(
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
        personal_residual_scale=m.get(
            "personal_residual_scale",
            0.20,
        ),
        use_digital_twin=spec["use_digital_twin"],
        use_moe=spec["use_moe"],
        use_personalization=spec["use_personalization"],
        router_mode=spec["router_mode"],
    )


def count_parameters(
    model,
):
    return int(
        sum(
            p.numel()
            for p in model.parameters()
        )
    )


def pooled_validation(
    client_models,
    val_loaders,
    device,
    cfg,
):
    shared_preds = []
    personal_preds = []
    targets = []
    residuals = []

    client_shared_rmse = []
    client_personal_rmse = []

    for model, loader in zip(
        client_models,
        val_loaders,
    ):
        pred = predict_v2(
            model,
            loader,
            device,
            cfg,
            collect_routing=False,
        )

        stats = metrics_from_prediction(
            pred
        )

        shared_preds.append(
            pred["shared_pred"]
        )
        personal_preds.append(
            pred["personal_pred"]
        )
        targets.append(
            pred["target"]
        )
        residuals.append(
            pred["residual_cycles"]
        )

        client_shared_rmse.append(
            stats["shared"]["rmse"]
        )
        client_personal_rmse.append(
            stats[
                "personalized"
            ]["rmse"]
        )

    pooled = {
        "shared_pred": torch.cat(
            shared_preds
        ),
        "personal_pred": torch.cat(
            personal_preds
        ),
        "target": torch.cat(
            targets
        ),
        "residual_cycles": torch.cat(
            residuals
        ),
    }

    stats = metrics_from_prediction(
        pooled
    )

    return {
        "shared_pooled_rmse": stats[
            "shared"
        ]["rmse"],
        "shared_pooled_mae": stats[
            "shared"
        ]["mae"],
        "shared_pooled_r2": stats[
            "shared"
        ]["r2"],
        "shared_pooled_nasa": stats[
            "shared"
        ]["nasa_score"],

        "personal_pooled_rmse": stats[
            "personalized"
        ]["rmse"],
        "personal_pooled_mae": stats[
            "personalized"
        ]["mae"],
        "personal_pooled_r2": stats[
            "personalized"
        ]["r2"],
        "personal_pooled_nasa": stats[
            "personalized"
        ]["nasa_score"],

        "personalization_gain_rmse": stats[
            "personalization_gain_rmse"
        ],

        "residual_abs_mean_cycles": stats[
            "residual_abs_mean_cycles"
        ],
        "residual_abs_p95_cycles": stats[
            "residual_abs_p95_cycles"
        ],
        "residual_abs_max_cycles": stats[
            "residual_abs_max_cycles"
        ],

        "mean_client_shared_rmse": float(
            np.mean(
                client_shared_rmse
            )
        ),
        "worst_client_shared_rmse": float(
            np.max(
                client_shared_rmse
            )
        ),
        "mean_client_personal_rmse": float(
            np.mean(
                client_personal_rmse
            )
        ),
        "worst_client_personal_rmse": float(
            np.max(
                client_personal_rmse
            )
        ),
    }


def save_official_predictions(
    method_dir,
    dataset,
    prediction,
):
    frame = pd.DataFrame({
        "engine_id": np.asarray(
            dataset.unit_ids
        ),
        "regime_id": np.asarray(
            dataset.regime_ids
        ),
        "true_rul": prediction[
            "target"
        ].numpy(),
        "shared_pred_rul": prediction[
            "shared_pred"
        ].numpy(),
    })

    frame[
        "error"
    ] = (
        frame["shared_pred_rul"]
        - frame["true_rul"]
    )

    frame[
        "abs_error"
    ] = np.abs(
        frame["error"]
    )

    frame.to_csv(
        method_dir
        / "official_test_predictions.csv",
        index=False,
    )


def resolve_baseline_suite(
    outputs_root,
    baseline_suite_arg,
):
    if baseline_suite_arg:
        path = Path(
            baseline_suite_arg
        )
    else:
        pointer = (
            outputs_root
            / "latest_benchmark.txt"
        )

        if not pointer.exists():
            return None

        path = Path(
            pointer.read_text(
                encoding="utf-8"
            ).strip()
        )

    if not path.exists():
        return None

    return path


def build_merged_comparison(
    suite_dir,
    baseline_suite,
):
    v2_rows = []

    for result_path in sorted(
        suite_dir.glob(
            "*/result.json"
        )
    ):
        v2_rows.append(
            json.loads(
                result_path.read_text(
                    encoding="utf-8"
                )
            )
        )

    v2_df = pd.DataFrame(
        v2_rows
    )

    v2_path = (
        suite_dir
        / "v2_results.csv"
    )

    v2_df.to_csv(
        v2_path,
        index=False,
    )

    if baseline_suite is None:
        return (
            v2_path,
            None,
        )

    baseline_csv = (
        baseline_suite
        / "benchmark_results.csv"
    )

    if not baseline_csv.exists():
        return (
            v2_path,
            None,
        )

    old = pd.read_csv(
        baseline_csv
    )

    baseline_rows = []

    for _, row in old.iterrows():
        baseline_rows.append({
            "method": row["method"],
            "category": row["category"],
            "description": row["description"],
            "version": "v1_fixed_baseline",
            "best_round": row[
                "best_round_or_epoch"
            ],
            "shared_val_rmse": row[
                "shared_val_rmse"
            ],
            "personal_val_rmse": row[
                "val_rmse"
            ],
            "personalization_gain_rmse": row[
                "personalization_gain_rmse"
            ],
            "test_rmse": row[
                "test_rmse"
            ],
            "test_mae": row[
                "test_mae"
            ],
            "test_r2": row[
                "test_r2"
            ],
            "test_nasa": row[
                "test_nasa"
            ],
            "train_seconds": row[
                "train_seconds"
            ],
            "parameters": row[
                "parameters"
            ],
        })

    merged = pd.concat(
        [
            pd.DataFrame(
                baseline_rows
            ),
            v2_df,
        ],
        ignore_index=True,
        sort=False,
    )

    merged_path = (
        suite_dir
        / "v2_vs_fixed_baselines.csv"
    )

    merged.to_csv(
        merged_path,
        index=False,
    )

    return (
        v2_path,
        merged_path,
    )


def run_method(
    method_name,
    spec,
    cfg,
    train_sets,
    val_sets,
    test_dataset,
    twin_aux_dim,
    device,
    method_dir,
):
    print(
        f"\n=== {method_name} ==="
    )

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
        spec,
        twin_aux_dim,
    ).cpu()

    client_models = [
        build_model(
            cfg,
            spec,
            twin_aux_dim,
        ).to(device)
        for _ in train_sets
    ]

    best_value = None
    best_round = None
    best_global_state = None
    best_local_states = None
    best_validation = None

    round_rows = []

    start = time.perf_counter()

    total_rounds = int(
        cfg["training"]["rounds"]
    )

    participation_rate = float(
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

        selected_count = max(
            1,
            int(
                round(
                    len(
                        client_models
                    )
                    * participation_rate
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
            selected_count,
        )

        client_states = []
        sample_counts = []
        expert_utils = []

        local_losses = []
        shared_rul_losses = []
        personal_rul_losses = []

        for cid in selected_ids:
            model = client_models[
                cid
            ]

            load_global_preserve_local(
                model,
                global_state,
            )

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg["training"]["lr"],
                weight_decay=cfg["training"][
                    "weight_decay"
                ],
            )

            result = train_local_v2(
                model,
                train_loaders[
                    cid
                ],
                optimizer,
                device,
                cfg,
                global_state=global_state,
            )

            client_states.append(
                clone_cpu_state(
                    model
                )
            )

            sample_counts.append(
                len(
                    train_sets[
                        cid
                    ]
                )
            )

            expert_utils.append(
                result[
                    "expert_utilization"
                ]
            )

            local_losses.append(
                result[
                    "total"
                ]
            )

            shared_rul_losses.append(
                result[
                    "shared_rul"
                ]
            )

            personal_rul_losses.append(
                result[
                    "personal_rul"
                ]
            )

        aggregated = aggregate_v2(
            client_states,
            sample_counts,
            expert_utils,
            mode=spec[
                "aggregation"
            ],
        )

        global_model.load_state_dict(
            aggregated,
            strict=True,
        )

        for model in (
            client_models
        ):
            load_global_preserve_local(
                model,
                aggregated,
            )

        validation = pooled_validation(
            client_models,
            val_loaders,
            device,
            cfg,
        )

        row = {
            "round": round_idx,
            "mean_total_loss": float(
                np.mean(
                    local_losses
                )
            ),
            "mean_shared_rul_loss": float(
                np.mean(
                    shared_rul_losses
                )
            ),
            "mean_personal_rul_loss": float(
                np.mean(
                    personal_rul_losses
                )
            ),
            **validation,
        }

        round_rows.append(
            row
        )

        selection_value = float(
            validation[
                "shared_pooled_rmse"
            ]
        )

        if (
            best_value is None
            or selection_value
            < best_value
        ):
            best_value = (
                selection_value
            )

            best_round = int(
                round_idx
            )

            best_global_state = (
                clone_cpu_state(
                    global_model
                )
            )

            best_local_states = {
                cid: local_state(
                    model
                )
                for cid, model in enumerate(
                    client_models
                )
            }

            best_validation = dict(
                validation
            )

            best_marker = " | NEW BEST"
        else:
            best_marker = ""

        if (
            round_idx == 1
            or round_idx % 5 == 0
            or round_idx
            == total_rounds
        ):
            print(
                f"  round "
                f"{round_idx:02d}/"
                f"{total_rounds} | "
                f"shared RMSE="
                f"{validation['shared_pooled_rmse']:.3f} | "
                f"personal RMSE="
                f"{validation['personal_pooled_rmse']:.3f} | "
                f"gain="
                f"{validation['personalization_gain_rmse']:+.3f} | "
                f"best shared="
                f"{best_value:.3f}"
                f"{best_marker}",
                flush=True,
            )

    train_seconds = (
        time.perf_counter()
        - start
    )

    global_model.load_state_dict(
        best_global_state,
        strict=True,
    )

    for cid, model in enumerate(
        client_models
    ):
        load_global_preserve_local(
            model,
            best_global_state,
        )

        restore_local_state(
            model,
            best_local_states[
                cid
            ],
        )

    selected_validation = (
        pooled_validation(
            client_models,
            val_loaders,
            device,
            cfg,
        )
    )

    # Official test: evaluate the directly supervised shared path only.
    test_model = build_model(
        cfg,
        spec,
        twin_aux_dim,
    )

    test_model.load_state_dict(
        best_global_state,
        strict=True,
    )

    test_model = test_model.to(
        device
    )

    test_prediction = predict_v2(
        test_model,
        test_loader,
        device,
        cfg,
        collect_routing=True,
    )

    test_metrics = metric_dict(
        test_prediction[
            "shared_pred"
        ],
        test_prediction[
            "target"
        ],
    )

    save_official_predictions(
        method_dir,
        test_dataset,
        test_prediction,
    )

    pd.DataFrame(
        round_rows
    ).to_csv(
        method_dir
        / "round_metrics.csv",
        index=False,
    )

    checkpoint = {
        "method": method_name,
        "method_spec": spec,
        "config": cfg,
        "best_round": best_round,
        "selection_metric": (
            "shared_pooled_rmse"
        ),
        "validation": (
            selected_validation
        ),
        "global_model_state": (
            best_global_state
        ),
        "client_local_states": (
            best_local_states
        ),
    }

    torch.save(
        checkpoint,
        method_dir
        / "best_model.pt",
    )

    result = {
        "method": method_name,
        "category": spec[
            "category"
        ],
        "description": spec[
            "description"
        ],
        "version": "v2_repair",
        "best_round": best_round,

        "shared_val_rmse": (
            selected_validation[
                "shared_pooled_rmse"
            ]
        ),
        "shared_val_mae": (
            selected_validation[
                "shared_pooled_mae"
            ]
        ),
        "shared_val_r2": (
            selected_validation[
                "shared_pooled_r2"
            ]
        ),
        "shared_val_nasa": (
            selected_validation[
                "shared_pooled_nasa"
            ]
        ),

        "personal_val_rmse": (
            selected_validation[
                "personal_pooled_rmse"
            ]
        ),
        "personal_val_mae": (
            selected_validation[
                "personal_pooled_mae"
            ]
        ),
        "personal_val_r2": (
            selected_validation[
                "personal_pooled_r2"
            ]
        ),

        "personalization_gain_rmse": (
            selected_validation[
                "personalization_gain_rmse"
            ]
        ),

        "mean_client_shared_rmse": (
            selected_validation[
                "mean_client_shared_rmse"
            ]
        ),
        "worst_client_shared_rmse": (
            selected_validation[
                "worst_client_shared_rmse"
            ]
        ),
        "mean_client_personal_rmse": (
            selected_validation[
                "mean_client_personal_rmse"
            ]
        ),
        "worst_client_personal_rmse": (
            selected_validation[
                "worst_client_personal_rmse"
            ]
        ),

        "residual_abs_mean_cycles": (
            selected_validation[
                "residual_abs_mean_cycles"
            ]
        ),
        "residual_abs_p95_cycles": (
            selected_validation[
                "residual_abs_p95_cycles"
            ]
        ),
        "residual_abs_max_cycles": (
            selected_validation[
                "residual_abs_max_cycles"
            ]
        ),

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
        "aggregation": spec[
            "aggregation"
        ],
        "uses_dt": bool(
            spec[
                "use_digital_twin"
            ]
        ),
        "uses_moe": bool(
            spec[
                "use_moe"
            ]
        ),
        "uses_personalization": bool(
            spec[
                "use_personalization"
            ]
        ),
    }

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

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=(
            "configs/v2_fd004.yaml"
        ),
    )

    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--suite-dir",
        default=None,
    )

    parser.add_argument(
        "--baseline-suite",
        default=None,
        help=(
            "Existing v1 benchmark directory. "
            "If omitted, outputs/latest_benchmark.txt is used."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
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
            "GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    train_dataset = (
        load_processed_cmapss(
            cfg["data"][
                "train_path"
            ]
        )
    )

    test_dataset = (
        load_processed_cmapss(
            cfg["data"][
                "test_path"
            ]
        )
    )

    twin_aux_dim = int(
        train_dataset.twin_aux.shape[
            -1
        ]
    )

    client_sets = (
        balanced_regime_partition(
            train_dataset,
            num_clients=cfg[
                "data"
            ]["num_clients"],
            alpha=cfg[
                "data"
            ][
                "non_iid_alpha"
            ],
            seed=cfg["seed"],
            affinity_weight=cfg[
                "data"
            ].get(
                "regime_affinity_weight",
                0.20,
            ),
        )
    )

    splits = [
        split_subset_by_unit(
            subset,
            val_fraction=cfg[
                "data"
            ].get(
                "val_fraction",
                0.2,
            ),
            seed=cfg["seed"] + i,
        )
        for i, subset in enumerate(
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
        f"{len(train_dataset):,}"
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

    requested = (
        args.methods
        if args.methods
        else cfg[
            "v2_suite"
        ]["methods"]
    )

    unknown = [
        x
        for x in requested
        if x not in METHODS
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
            / f"v2_repair_FD004_{stamp}"
        )

    suite_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        output_root
        / "latest_v2.txt"
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

    baseline_suite = (
        resolve_baseline_suite(
            output_root,
            args.baseline_suite,
        )
    )

    if baseline_suite:
        print(
            f"Fixed v1 baseline suite: "
            f"{baseline_suite}"
        )
    else:
        print(
            "Warning: no fixed baseline suite found. "
            "V2 experiments will still run."
        )

    suite_start = time.perf_counter()

    for method_name in requested:
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

        spec = METHODS[
            method_name
        ]

        this_cfg = method_cfg(
            cfg,
            spec,
        )

        result = run_method(
            method_name,
            spec,
            this_cfg,
            train_sets,
            val_sets,
            test_dataset,
            twin_aux_dim,
            device,
            method_dir,
        )

        v2_path, merged_path = (
            build_merged_comparison(
                suite_dir,
                baseline_suite,
            )
        )

        print(
            f"  completed "
            f"{method_name} | "
            f"shared val RMSE="
            f"{result['shared_val_rmse']:.3f} | "
            f"personal val RMSE="
            f"{result['personal_val_rmse']:.3f} | "
            f"test RMSE="
            f"{result['test_rmse']:.3f}"
        )

        print(
            f"  current V2 table: "
            f"{v2_path}"
        )

        if merged_path:
            print(
                f"  merged comparison: "
                f"{merged_path}"
            )

    elapsed = (
        time.perf_counter()
        - suite_start
    )

    v2_path, merged_path = (
        build_merged_comparison(
            suite_dir,
            baseline_suite,
        )
    )

    summary = {
        "suite_dir": str(
            suite_dir
        ),
        "seed": int(
            cfg["seed"]
        ),
        "methods": requested,
        "selection_metric": (
            "shared_pooled_rmse"
        ),
        "baseline_suite": (
            str(
                baseline_suite
            )
            if baseline_suite
            else None
        ),
        "v2_results_csv": str(
            v2_path
        ),
        "merged_results_csv": (
            str(
                merged_path
            )
            if merged_path
            else None
        ),
        "elapsed_seconds_this_invocation": float(
            elapsed
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
        "\nV2 repair suite complete."
    )

    print(
        f"V2 results: "
        f"{v2_path}"
    )

    if merged_path:
        print(
            f"V2 + fixed baselines: "
            f"{merged_path}"
        )

    print(
        f"Elapsed: "
        f"{elapsed / 60.0:.2f} min"
    )


if __name__ == "__main__":
    main()
