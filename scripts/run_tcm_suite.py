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
from twinfed_moe.federated.tcm_aggregation import (
    fedavg_state_dicts,
)
from twinfed_moe.models.tcm_model import (
    TwinFedTCM,
)
from twinfed_moe.training.metrics import (
    metric_dict,
)
from twinfed_moe.training.tcm_trainer import (
    predict_tcm,
    train_local_tcm,
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
    "dt_concat_fedavg": {
        "category": "fusion_baseline",
        "fusion_mode": "concat",
        "aggregation": "fedavg",
        "prox_mu": 0.0,
        "use_dt_reconstruction": True,
        "lambda_dt": 0.10,
        "description": (
            "Digital-Twin state concatenated with the Transformer "
            "representation, trained with FedAvg."
        ),
    },

    "tcm_fedavg": {
        "category": "proposed_candidate",
        "fusion_mode": "tcm",
        "aggregation": "fedavg",
        "prox_mu": 0.0,
        "use_dt_reconstruction": True,
        "lambda_dt": 0.10,
        "description": (
            "Twin-Conditioned Modulation of Transformer features "
            "with FedAvg."
        ),
    },

    "tcm_fedprox": {
        "category": "proposed_candidate",
        "fusion_mode": "tcm",
        "aggregation": "fedavg",
        "prox_mu": 0.001,
        "use_dt_reconstruction": True,
        "lambda_dt": 0.10,
        "description": (
            "Twin-Conditioned Modulation of Transformer features "
            "with FedProx."
        ),
    },

    "tcm_no_dt_reconstruction": {
        "category": "ablation",
        "fusion_mode": "tcm",
        "aggregation": "fedavg",
        "prox_mu": 0.001,
        "use_dt_reconstruction": False,
        "lambda_dt": 0.0,
        "description": (
            "TCM + FedProx without the Digital-Twin reconstruction "
            "auxiliary objective."
        ),
    },

    "tcm_identity_modulation": {
        "category": "ablation",
        "fusion_mode": "identity",
        "aggregation": "fedavg",
        "prox_mu": 0.001,
        "use_dt_reconstruction": True,
        "lambda_dt": 0.10,
        "description": (
            "FedProx Transformer with the same Digital-Twin auxiliary "
            "branch but no direct twin-conditioned feature modulation."
        ),
    },
}


def clone_cpu_state(
    model,
):
    return {
        k: v.detach().cpu().clone()
        for k, v
        in model.state_dict().items()
    }


def make_loader(
    dataset,
    cfg,
    shuffle=False,
):
    return DataLoader(
        dataset,
        batch_size=cfg[
            "data"
        ][
            "batch_size"
        ],
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

    cfg[
        "training"
    ][
        "prox_mu"
    ] = float(
        spec[
            "prox_mu"
        ]
    )

    cfg[
        "training"
    ][
        "lambda_dt"
    ] = float(
        spec[
            "lambda_dt"
        ]
    )

    return cfg


def build_model(
    cfg,
    spec,
    twin_aux_dim,
):
    m = cfg[
        "model"
    ]

    return TwinFedTCM(
        num_sensors=cfg[
            "data"
        ][
            "num_sensors"
        ],
        twin_aux_dim=twin_aux_dim,
        seq_len=cfg[
            "data"
        ][
            "seq_len"
        ],
        d_model=m[
            "d_model"
        ],
        nhead=m[
            "nhead"
        ],
        transformer_layers=m[
            "transformer_layers"
        ],
        dim_feedforward=m[
            "dim_feedforward"
        ],
        twin_latent_dim=m[
            "twin_latent_dim"
        ],
        fusion_hidden_dim=m[
            "fusion_hidden_dim"
        ],
        dropout=m[
            "dropout"
        ],
        fusion_mode=spec[
            "fusion_mode"
        ],
        use_dt_reconstruction=spec[
            "use_dt_reconstruction"
        ],
    )


def count_parameters(
    model,
):
    return int(
        sum(
            p.numel()
            for p
            in model.parameters()
        )
    )


def pooled_validation(
    models,
    loaders,
    device,
    cfg,
):
    preds = []
    targets = []
    per_client = []

    for model, loader in zip(
        models,
        loaders,
    ):
        result = predict_tcm(
            model,
            loader,
            device,
            cfg,
            collect_modulation=False,
        )

        metrics = metric_dict(
            result[
                "pred"
            ],
            result[
                "target"
            ],
        )

        preds.append(
            result[
                "pred"
            ]
        )

        targets.append(
            result[
                "target"
            ]
        )

        per_client.append(
            metrics
        )

    pooled_pred = torch.cat(
        preds
    )

    pooled_target = torch.cat(
        targets
    )

    pooled = metric_dict(
        pooled_pred,
        pooled_target,
    )

    return {
        "pooled_rmse": pooled[
            "rmse"
        ],
        "pooled_mae": pooled[
            "mae"
        ],
        "pooled_r2": pooled[
            "r2"
        ],
        "pooled_nasa": pooled[
            "nasa_score"
        ],
        "mean_client_rmse": float(
            np.mean(
                [
                    x[
                        "rmse"
                    ]
                    for x
                    in per_client
                ]
            )
        ),
        "worst_client_rmse": float(
            np.max(
                [
                    x[
                        "rmse"
                    ]
                    for x
                    in per_client
                ]
            )
        ),
    }


def save_test_diagnostics(
    method_dir,
    dataset,
    prediction,
):
    pred = prediction[
        "pred"
    ].numpy()

    target = prediction[
        "target"
    ].numpy()

    frame = pd.DataFrame({
        "engine_id": np.asarray(
            dataset.unit_ids
        ),
        "regime_id": np.asarray(
            dataset.regime_ids
        ),
        "true_rul": target,
        "pred_rul": pred,
        "error": (
            pred
            - target
        ),
        "abs_error": np.abs(
            pred
            - target
        ),
        "gamma_abs_shift": (
            prediction[
                "gamma_abs_shift"
            ].numpy()
        ),
        "beta_abs_mean": (
            prediction[
                "beta_abs_mean"
            ].numpy()
        ),
        "twin_state_norm": (
            prediction[
                "twin_state_norm"
            ].numpy()
        ),
    })

    frame.to_csv(
        method_dir
        / "official_test_predictions.csv",
        index=False,
    )

    regime_rows = []

    for regime in sorted(
        frame[
            "regime_id"
        ].unique()
    ):
        subset = frame[
            frame[
                "regime_id"
            ] == regime
        ]

        p = torch.as_tensor(
            subset[
                "pred_rul"
            ].to_numpy(),
            dtype=torch.float32,
        )

        y = torch.as_tensor(
            subset[
                "true_rul"
            ].to_numpy(),
            dtype=torch.float32,
        )

        metrics = metric_dict(
            p,
            y,
        )

        regime_rows.append({
            "regime_id": int(
                regime
            ),
            "count": int(
                len(
                    subset
                )
            ),
            **metrics,
            "gamma_abs_shift": float(
                subset[
                    "gamma_abs_shift"
                ].mean()
            ),
            "beta_abs_mean": float(
                subset[
                    "beta_abs_mean"
                ].mean()
            ),
            "twin_state_norm": float(
                subset[
                    "twin_state_norm"
                ].mean()
            ),
        })

    pd.DataFrame(
        regime_rows
    ).to_csv(
        method_dir
        / "regime_diagnostics.csv",
        index=False,
    )

    return {
        "gamma_abs_shift_mean": float(
            frame[
                "gamma_abs_shift"
            ].mean()
        ),
        "gamma_abs_shift_p95": float(
            frame[
                "gamma_abs_shift"
            ].quantile(
                0.95
            )
        ),
        "beta_abs_mean_mean": float(
            frame[
                "beta_abs_mean"
            ].mean()
        ),
        "beta_abs_mean_p95": float(
            frame[
                "beta_abs_mean"
            ].quantile(
                0.95
            )
        ),
        "twin_state_norm_mean": float(
            frame[
                "twin_state_norm"
            ].mean()
        ),
    }


def resolve_baseline_suite(
    output_root,
    baseline_suite_arg,
):
    if baseline_suite_arg:
        path = Path(
            baseline_suite_arg
        )
    else:
        pointer = (
            output_root
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
    tcm_rows = []

    for result_path in sorted(
        suite_dir.glob(
            "*/result.json"
        )
    ):
        tcm_rows.append(
            json.loads(
                result_path.read_text(
                    encoding="utf-8"
                )
            )
        )

    tcm_df = pd.DataFrame(
        tcm_rows
    )

    tcm_path = (
        suite_dir
        / "tcm_results.csv"
    )

    tcm_df.to_csv(
        tcm_path,
        index=False,
    )

    if baseline_suite is None:
        return (
            tcm_path,
            None,
        )

    baseline_csv = (
        baseline_suite
        / "benchmark_results.csv"
    )

    if not baseline_csv.exists():
        return (
            tcm_path,
            None,
        )

    baseline = pd.read_csv(
        baseline_csv
    )

    keep_methods = {
        "centralized_transformer",
        "local_only_transformer",
        "fedavg_transformer",
        "fedprox_transformer",
        "fedavg_moe",
    }

    baseline = baseline[
        baseline[
            "method"
        ].isin(
            keep_methods
        )
    ].copy()

    baseline_rows = []

    for _, row in (
        baseline.iterrows()
    ):
        baseline_rows.append({
            "method": row[
                "method"
            ],
            "category": row[
                "category"
            ],
            "description": row[
                "description"
            ],
            "version": (
                "fixed_v1_baseline"
            ),
            "best_round": row[
                "best_round_or_epoch"
            ],
            "val_rmse": row[
                "val_rmse"
            ],
            "val_mae": row[
                "val_mae"
            ],
            "val_r2": row[
                "val_r2"
            ],
            "val_nasa": row[
                "val_nasa"
            ],
            "mean_client_rmse": row[
                "mean_client_rmse"
            ],
            "worst_client_rmse": row[
                "worst_client_rmse"
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
            tcm_df,
        ],
        ignore_index=True,
        sort=False,
    )

    merged_path = (
        suite_dir
        / "tcm_vs_fixed_baselines.csv"
    )

    merged.to_csv(
        merged_path,
        index=False,
    )

    return (
        tcm_path,
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
        ).to(
            device
        )
        for _ in train_sets
    ]

    best_rmse = None
    best_round = None
    best_state = None
    best_validation = None

    round_rows = []

    run_start = (
        time.perf_counter()
    )

    total_rounds = int(
        cfg[
            "training"
        ][
            "rounds"
        ]
    )

    participation = float(
        cfg[
            "federated"
        ][
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
            selected_count,
        )

        states = []
        sample_counts = []
        losses = []
        rul_losses = []
        dt_losses = []
        prox_losses = []

        for cid in selected_ids:
            model = client_models[
                cid
            ]

            model.load_state_dict(
                global_state,
                strict=True,
            )

            optimizer = (
                torch.optim.AdamW(
                    model.parameters(),
                    lr=cfg[
                        "training"
                    ][
                        "lr"
                    ],
                    weight_decay=cfg[
                        "training"
                    ][
                        "weight_decay"
                    ],
                )
            )

            result = train_local_tcm(
                model,
                train_loaders[
                    cid
                ],
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
                    train_sets[
                        cid
                    ]
                )
            )

            losses.append(
                result[
                    "loss"
                ]
            )

            rul_losses.append(
                result[
                    "rul_loss"
                ]
            )

            dt_losses.append(
                result[
                    "dt_loss"
                ]
            )

            prox_losses.append(
                result[
                    "prox_loss"
                ]
            )

        aggregated = (
            fedavg_state_dicts(
                states,
                sample_counts,
            )
        )

        global_model.load_state_dict(
            aggregated,
            strict=True,
        )

        for model in (
            client_models
        ):
            model.load_state_dict(
                aggregated,
                strict=True,
            )

        validation = pooled_validation(
            client_models,
            val_loaders,
            device,
            cfg,
        )

        row = {
            "round": round_idx,
            "mean_loss": float(
                np.mean(
                    losses
                )
            ),
            "mean_rul_loss": float(
                np.mean(
                    rul_losses
                )
            ),
            "mean_dt_loss": float(
                np.mean(
                    dt_losses
                )
            ),
            "mean_prox_loss": float(
                np.mean(
                    prox_losses
                )
            ),
            **validation,
        }

        round_rows.append(
            row
        )

        current_rmse = float(
            validation[
                "pooled_rmse"
            ]
        )

        if (
            best_rmse is None
            or current_rmse
            < best_rmse
        ):
            best_rmse = (
                current_rmse
            )
            best_round = int(
                round_idx
            )
            best_state = (
                clone_cpu_state(
                    global_model
                )
            )
            best_validation = dict(
                validation
            )
            marker = (
                " | NEW BEST"
            )
        else:
            marker = ""

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
                f"val RMSE="
                f"{current_rmse:.3f} | "
                f"best="
                f"{best_rmse:.3f}"
                f"{marker}",
                flush=True,
            )

    train_seconds = (
        time.perf_counter()
        - run_start
    )

    global_model.load_state_dict(
        best_state,
        strict=True,
    )

    test_model = build_model(
        cfg,
        spec,
        twin_aux_dim,
    )

    test_model.load_state_dict(
        best_state,
        strict=True,
    )

    test_model = test_model.to(
        device
    )

    test_prediction = predict_tcm(
        test_model,
        test_loader,
        device,
        cfg,
        collect_modulation=True,
    )

    test_metrics = metric_dict(
        test_prediction[
            "pred"
        ],
        test_prediction[
            "target"
        ],
    )

    diagnostics = (
        save_test_diagnostics(
            method_dir,
            test_dataset,
            test_prediction,
        )
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
        "validation": (
            best_validation
        ),
        "global_model_state": (
            best_state
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
        "version": "tcm_v1",
        "best_round": best_round,

        "val_rmse": (
            best_validation[
                "pooled_rmse"
            ]
        ),
        "val_mae": (
            best_validation[
                "pooled_mae"
            ]
        ),
        "val_r2": (
            best_validation[
                "pooled_r2"
            ]
        ),
        "val_nasa": (
            best_validation[
                "pooled_nasa"
            ]
        ),
        "mean_client_rmse": (
            best_validation[
                "mean_client_rmse"
            ]
        ),
        "worst_client_rmse": (
            best_validation[
                "worst_client_rmse"
            ]
        ),

        "test_rmse": (
            test_metrics[
                "rmse"
            ]
        ),
        "test_mae": (
            test_metrics[
                "mae"
            ]
        ),
        "test_r2": (
            test_metrics[
                "r2"
            ]
        ),
        "test_nasa": (
            test_metrics[
                "nasa_score"
            ]
        ),

        **diagnostics,

        "fusion_mode": spec[
            "fusion_mode"
        ],
        "prox_mu": float(
            spec[
                "prox_mu"
            ]
        ),
        "dt_reconstruction": bool(
            spec[
                "use_dt_reconstruction"
            ]
        ),
        "lambda_dt": float(
            spec[
                "lambda_dt"
            ]
        ),
        "train_seconds": float(
            train_seconds
        ),
        "parameters": (
            count_parameters(
                test_model
            )
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
        default="configs/tcm_fd004.yaml",
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
            "Existing fixed benchmark directory. "
            "Default: outputs/latest_benchmark.txt"
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
            ][
                "num_clients"
            ],
            alpha=cfg[
                "data"
            ][
                "non_iid_alpha"
            ],
            seed=cfg[
                "seed"
            ],
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
            seed=cfg[
                "seed"
            ] + i,
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
            "tcm_suite"
        ][
            "methods"
        ]
    )

    unknown = [
        name
        for name in requested
        if name not in METHODS
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
        stamp = (
            datetime.now()
            .strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        suite_dir = (
            output_root
            / f"tcm_FD004_{stamp}"
        )

    suite_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        output_root
        / "latest_tcm.txt"
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
            f"Fixed baseline suite: "
            f"{baseline_suite}"
        )
    else:
        print(
            "Warning: fixed baseline suite "
            "not found. TCM methods will "
            "still run."
        )

    start = time.perf_counter()

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

        # Deterministic method reset.
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

        tcm_path, merged_path = (
            build_merged_comparison(
                suite_dir,
                baseline_suite,
            )
        )

        print(
            f"  completed "
            f"{method_name} | "
            f"val RMSE="
            f"{result['val_rmse']:.3f} | "
            f"test RMSE="
            f"{result['test_rmse']:.3f}"
        )

        print(
            f"  current TCM table: "
            f"{tcm_path}"
        )

        if merged_path:
            print(
                f"  merged comparison: "
                f"{merged_path}"
            )

    elapsed = (
        time.perf_counter()
        - start
    )

    tcm_path, merged_path = (
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
            "validation pooled RMSE"
        ),
        "baseline_suite": (
            str(
                baseline_suite
            )
            if baseline_suite
            else None
        ),
        "tcm_results_csv": str(
            tcm_path
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
        "\nTwinFed-TCM suite complete."
    )

    print(
        f"TCM results: "
        f"{tcm_path}"
    )

    if merged_path:
        print(
            f"TCM + fixed baselines: "
            f"{merged_path}"
        )

    print(
        f"Elapsed: "
        f"{elapsed / 60.0:.2f} min"
    )


if __name__ == "__main__":
    main()
